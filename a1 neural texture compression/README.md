# Neural Texture Compression

P1: code in `a1_p1p2.py`.

P2: code in `a1_p1p2.py`. To run, `uv run a1_p1p2.py`. This prints the PSNRs and saves the compressed textures to `s3tc_compressed`.

P3-P5: code in `a1_p3p4p5.py`. To run, `uv run a1_p3p4p5.py`. This prints the PSNRs and saves the compressed textures to `mlp_large_compressed` or other similar folders. Models are stored in `models`. Output is manually stored in `final_results.txt`.

P6: graphs are plotted with `plot_size_psnr.py` and `plot_psnr.py`. Plots were formatted with the help of Claude Code. Plots are saved in `figures`.

P7: code in `quantize.py`. To run, `uv run quantize.py`. This quantizes the models, prints out PSNR and compression ratio metrics, and saves compressed textures to `mlp_large_compressed_quantized` or other similar folders. Quantized models are not saved.


