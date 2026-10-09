"""
Interactive viewer for a trained set of 3D Gaussians (saved by a2_3D.py as gaussians.pt).

    python view_gaussians.py                       # newest images/*/gaussians.pt
    python view_gaussians.py --ckpt path/to/gaussians.pt
    python view_gaussians.py --snapshot out.png    # render the initial view to a file and exit

Controls
    left-drag   orbit (azimuth / elevation)
    scroll      zoom (orbit radius)
    n           snap to the nearest train/val camera (shows PSNR against its image)
    r           reset to training camera 0
    s           save the current render as a PNG
    q           quit

The right panel always shows the ground-truth image of the nearest dataset camera,
so floaters and view-dependent artifacts are easy to spot while orbiting.

The math (quaternion -> rotation, 3D covariance, projection, compositing) mirrors a2_3D.py.
The Gaussian weight uses the closed-form 2x2 inverse, which is identical to the matmul
version to ~1e-7 and several times faster.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
from PIL import Image

SIG2_DILATION = 0.0   # set to 0.3 if you add the 0.3 px dilation to project_gaussian in a2_3D.py


# ----------------------------------------------------------------------------- math (mirrors a2_3D.py)
def get_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def quaternion_to_rotation(q):
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def covariance_3d(scale, quat):
    R = quaternion_to_rotation(quat)
    S = torch.diag_embed(scale)
    return R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)


def project_gaussian(mu3, Sigma3, R_wc, t, K):
    mu_cam = mu3 @ R_wc.T + t
    z = mu_cam[:, 2]
    mu2 = (mu_cam @ K.T)[:, :2] / z[:, None]
    J = torch.zeros((mu_cam.shape[0], 2, 3), device=mu_cam.device)
    J[:, 0, 0] = K[0, 0] / z
    J[:, 0, 2] = -K[0, 0] * mu_cam[:, 0] / (z * z)
    J[:, 1, 1] = K[1, 1] / z
    J[:, 1, 2] = -K[1, 1] * mu_cam[:, 1] / (z * z)
    Scam = R_wc @ Sigma3 @ R_wc.T
    Sig2 = J @ Scam @ J.transpose(-1, -2)
    if SIG2_DILATION > 0:
        Sig2 = Sig2 + SIG2_DILATION * torch.eye(2, device=Sig2.device)
    return mu2, Sig2, z


def gaussian_weight(xy, mu, Sigma):
    a, b, d = Sigma[:, 0, 0], Sigma[:, 0, 1], Sigma[:, 1, 1]
    det = a * d - b * b
    ia, ib, id_ = d / det, -b / det, a / det
    dx = xy[:, 0:1] - mu[None, :, 0]
    dy = xy[:, 1:2] - mu[None, :, 1]
    return torch.exp(-0.5 * (ia * dx * dx + 2 * ib * dx * dy + id_ * dy * dy))


def render(mu, Sigma, color, opacity, order, H, W):
    device, dtype = mu.device, mu.dtype
    xs = torch.arange(W, device=device, dtype=dtype)
    ys = torch.arange(H, device=device, dtype=dtype)
    xy = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1).reshape(-1, 2)
    alpha = (opacity[None, :] * gaussian_weight(xy, mu, Sigma))[:, order]
    T = torch.cumprod(1 - alpha, dim=1)
    T = torch.cat([torch.ones_like(T[:, :1]), T[:, :-1]], dim=1)
    C = (T * alpha) @ color[order]
    return C.reshape(H, W, 3)


# ----------------------------------------------------------------------------- scene + camera
class GaussianScene:
    """Holds activated Gaussian parameters; the 3D covariance is view independent so it is built once."""

    def __init__(self, ckpt_path, device):
        ck = torch.load(ckpt_path, map_location="cpu")
        self.mu3 = ck["mu3"].float().to(device)
        self.Sig3 = covariance_3d(ck["log_s"].float().exp(), ck["quat"].float()).to(device)
        self.color = ck["color"].float().sigmoid().to(device)
        self.opacity = ck["op_raw"].float().sigmoid().to(device)
        self.meta = ck.get("meta", {})
        self.device = device

    @torch.no_grad()
    def render(self, R_wc, t, K, H, W):
        mu2, Sig2, depth = project_gaussian(self.mu3, self.Sig3, R_wc, t, K)
        order = torch.argsort(depth)
        return render(mu2, Sig2, self.color, self.opacity, order, H, W).clamp(0, 1)


def look_at(center, target, up=(0.0, 1.0, 0.0)):
    """OpenCV camera (x right, y down, z forward). Matches the dataset's R_wc / t convention exactly."""
    center = np.asarray(center, dtype=np.float64)
    fwd = target - center
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, dtype=np.float64))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R_wc = np.stack([right, down, fwd])
    t = -R_wc @ center
    return R_wc, t


def cam_center(frame):
    R = np.array(frame["R_wc"])
    t = np.array(frame["t"])
    return -R.T @ t


class OrbitCamera:
    def __init__(self, center, target=np.zeros(3)):
        self.target = np.asarray(target, dtype=np.float64)
        self.set_center(center)

    def set_center(self, center):
        v = np.asarray(center, dtype=np.float64) - self.target
        self.radius = float(np.linalg.norm(v))
        self.el = math.degrees(math.asin(np.clip(v[1] / self.radius, -1, 1)))
        self.az = math.degrees(math.atan2(v[0], v[2]))

    def center(self):
        el, az = math.radians(self.el), math.radians(self.az)
        return self.target + self.radius * np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])

    def pose(self, device):
        R_wc, t = look_at(self.center(), self.target)
        return (torch.tensor(R_wc, dtype=torch.float32, device=device),
                torch.tensor(t, dtype=torch.float32, device=device))


# ----------------------------------------------------------------------------- viewer
class Viewer:
    def __init__(self, scene, camera_data, root, res_scale=1.0, drag_scale=0.5):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.scene = scene
        self.device = scene.device
        self.root = root
        self.W0, self.H0 = camera_data["width"], camera_data["height"]
        self.K0 = np.array(camera_data["K"], dtype=np.float64)
        self.res_scale, self.drag_scale = res_scale, drag_scale
        self.frames = [dict(f, split="train") for f in camera_data["frames"]] + \
                      [dict(f, split="val") for f in camera_data["val_frames"]]
        self.centers = np.stack([cam_center(f) for f in self.frames])
        self.dirs = self.centers / np.linalg.norm(self.centers, axis=1, keepdims=True)
        self.gt_cache = {}
        self.cam = OrbitCamera(self.centers[0])
        self.drag_start = None

        self.fig, (self.ax_r, self.ax_gt) = plt.subplots(1, 2, figsize=(9, 4.8))
        for ax in (self.ax_r, self.ax_gt):
            ax.set_xticks([]); ax.set_yticks([])
        blank = np.zeros((self.H0, self.W0, 3))
        self.im_r = self.ax_r.imshow(blank, interpolation="nearest")
        self.im_gt = self.ax_gt.imshow(blank, interpolation="nearest")
        self.fig.text(0.5, 0.01, "drag: orbit   scroll: zoom   n: snap to nearest camera   r: reset   s: save   q: quit",
                      ha="center", va="bottom", fontsize=9)
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.last_img = None
        self.update(full=True)

    # -- rendering
    def render_current(self, scale):
        H, W = max(8, int(round(self.H0 * scale))), max(8, int(round(self.W0 * scale)))
        K = torch.tensor(self.K0 * np.array([[W / self.W0, 1, W / self.W0], [1, H / self.H0, H / self.H0], [1, 1, 1]]),
                         dtype=torch.float32, device=self.device)
        R_wc, t = self.cam.pose(self.device)
        img = self.scene.render(R_wc, t, K, H, W)
        if (H, W) != (self.H0, self.W0):
            img = torch.nn.functional.interpolate(img.permute(2, 0, 1)[None], size=(self.H0, self.W0), mode="nearest")[0].permute(1, 2, 0)
        return img.cpu().numpy()

    def nearest_frame(self):
        d = self.cam.center()
        d = d / np.linalg.norm(d)
        i = int(np.argmax(self.dirs @ d))
        ang = math.degrees(math.acos(float(np.clip(self.dirs[i] @ d, -1, 1))))
        return i, ang

    def gt_image(self, i):
        if i not in self.gt_cache:
            f = self.frames[i]
            self.gt_cache[i] = np.asarray(Image.open(os.path.join(self.root, f["file"])).convert("RGB")).astype(np.float32) / 255.0
        return self.gt_cache[i]

    def update(self, full):
        t0 = time.time()
        img = self.render_current(self.res_scale if full else self.res_scale * self.drag_scale)
        dt = time.time() - t0
        self.last_img = img
        i, ang = self.nearest_frame()
        gt = self.gt_image(i)
        self.im_r.set_data(img)
        self.im_gt.set_data(gt)
        n = self.scene.mu3.shape[0]
        self.ax_r.set_title(f"render  N={n}  az {self.cam.az:6.1f}°  el {self.cam.el:5.1f}°  r {self.cam.radius:.2f}   {1/dt:4.1f} fps", fontsize=9)
        psnr = ""
        if ang < 0.5 and abs(np.linalg.norm(self.centers[i]) - self.cam.radius) < 1e-3:
            mse = float(((img - gt) ** 2).mean())
            psnr = f"   PSNR {-10 * math.log10(max(mse, 1e-12)):.2f} dB"
        self.ax_gt.set_title(f"nearest {self.frames[i]['split']} camera: {self.frames[i]['file']}  ({ang:.1f}° away){psnr}", fontsize=9)
        self.fig.canvas.draw_idle()

    # -- events
    def on_press(self, ev):
        if ev.inaxes is self.ax_r and ev.button == 1:
            self.drag_start = (ev.x, ev.y, self.cam.az, self.cam.el)

    def on_release(self, ev):
        if self.drag_start is not None:
            self.drag_start = None
            self.update(full=True)

    def on_motion(self, ev):
        if self.drag_start is None:
            return
        x0, y0, az0, el0 = self.drag_start
        self.cam.az = az0 - 0.5 * (ev.x - x0)
        self.cam.el = float(np.clip(el0 + 0.5 * (ev.y - y0), -89.0, 89.0))
        self.update(full=False)

    def on_scroll(self, ev):
        self.cam.radius = float(np.clip(self.cam.radius * (0.9 if ev.button == "up" else 1.1), 0.5, 50.0))
        self.update(full=True)

    def on_key(self, ev):
        if ev.key == "r":
            self.cam = OrbitCamera(self.centers[0]); self.update(full=True)
        elif ev.key == "n":
            i, _ = self.nearest_frame()
            self.cam.set_center(self.centers[i]); self.update(full=True)
        elif ev.key == "s":
            out = f"view_az{self.cam.az:.0f}_el{self.cam.el:.0f}.png"
            Image.fromarray((self.last_img * 255).astype(np.uint8)).save(out); print(f"saved {out}")
        elif ev.key == "q":
            self.plt.close(self.fig)

    def show(self):
        self.plt.show()


# ----------------------------------------------------------------------------- main
def newest_checkpoint():
    paths = glob.glob("images/*/gaussians.pt")
    if not paths:
        raise SystemExit("no images/*/gaussians.pt found; pass --ckpt")
    return max(paths, key=os.path.getmtime)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="gaussians.pt saved by a2_3D.py (default: newest under images/)")
    p.add_argument("--cameras", default="images/spheres/cameras.json")
    p.add_argument("--res", type=float, default=1.0, help="render resolution multiplier (1.0 = dataset size)")
    p.add_argument("--drag-res", type=float, default=0.5, help="extra resolution multiplier while dragging")
    p.add_argument("--snapshot", default=None, help="render training camera 0 to this PNG and exit")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = get_device(args.cpu)
    ckpt = args.ckpt or newest_checkpoint()
    scene = GaussianScene(ckpt, device)
    camera_data = json.load(open(args.cameras))
    root = os.path.dirname(args.cameras)
    print(f"loaded {ckpt}: N={scene.mu3.shape[0]} meta={scene.meta} device={device}")

    if args.snapshot:
        import matplotlib
        matplotlib.use("Agg")
        v = Viewer(scene, camera_data, root, res_scale=args.res, drag_scale=args.drag_res)
        Image.fromarray((v.last_img * 255).astype(np.uint8)).save(args.snapshot)
        print(f"wrote {args.snapshot}")
        return

    Viewer(scene, camera_data, root, res_scale=args.res, drag_scale=args.drag_res).show()


if __name__ == "__main__":
    main()
