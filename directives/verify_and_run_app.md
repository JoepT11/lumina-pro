# Directive: Lumina Suite Verification & Quality Assurance

## 1. Goal
Systematically verify, test, and ensure error-free operation of the **Lumina Pro** AI Image Suite across all four functional modules:
1. ✂️ AI Background Remover (BiRefNet State-of-the-Art ONNX, alpha matting, feathering, model selection)
2. 🎞️ Cinematic 3D LUT Generator (Reinhard & perceptual grading, live video frame preview, 33x33x33 & 65x65x65 .cube LUT export voor DaVinci Resolve & Premiere Pro)
3. 📐 Smart Batch Resizer (smart crop, letterboxing, percentage scaling, Lanczos/Bicubic resampling)
4. 💧 Watermark & Branding (text/logo watermarking, 9-grid positioning, tiled mosaic, blend modes)

## 2. Inputs
- Python Environment: Python 3.12+ with `streamlit`, `rembg`, `pillow`, `opencv-python-headless`, `numpy`, `onnxruntime`.
- Test inputs: in-memory simulated uploaded files or test files in `.tmp/`.

## 3. Execution Tools
- Run the full test suite:
  ```powershell
  python execution/verify_app.py
  ```
- Run the Streamlit web application:
  ```powershell
  streamlit run app.py --server.port 8501 --server.headless true
  ```

## 4. Expected Outputs & Deliverables
- Clean test execution report with 0 failures:
  ```
  [1/4] Verifying Module 1: AI Background Remover... ✔ PASSED
  [2/4] Verifying Module 2: Cinematic LUT Studio...   ✔ PASSED
  [3/4] Verifying Module 3: Smart Batch Resizer...   ✔ PASSED
  [4/4] Verifying Module 4: Watermark & Branding...  ✔ PASSED
  ```
- Live web app accessible at `http://localhost:8501`.

## 5. Edge Cases & Learnings (Self-Annealing)
1. **Keyword Parameter Mismatch in `apply_watermark`**:
   - *Problem*: The UI called `apply_watermark(base_image=..., watermark_image=...)`, but the function signature was declared with `watermark_item`. This triggered an unexpected keyword argument `TypeError` during live preview and batch export.
   - *Fix*: Declared `watermark_image` as the primary argument with `watermark_item` fallback in the signature.
2. **RGBA to JPEG Pillow Saving (`OSError: cannot write mode RGBA as JPEG`)**:
   - *Problem*: When watermarking or resizing PNG files with alpha transparency and exporting as JPEG, Pillow raises an OSError.
   - *Fix*: Detect RGBA/transparency, save as `.png` if the source was PNG, or composite onto a clean white canvas (`(255, 255, 255)`) before saving as JPEG.
3. **Streamlit UI State Loss on Download Button Click**:
   - *Problem*: In Streamlit, clicking `st.download_button` triggers a script rerun. If results are only stored in local variables inside `if st.button(...)`, the download button and preview instantly vanish on click.
   - *Fix*: Persist processed ZIPs, previews, and counts in `st.session_state` (`t1_results`, `t2_results`, `t3_results`, `t4_results`).
4. **Cinematic 3D LUT Color Space Calibration**:
   - *Problem*: The 3D lattice in `generate_3d_cube_lut` was passed without target sample statistics, leading to incorrect color grading inside DaVinci Resolve / Premiere Pro.
   - *Fix*: Compute CIELAB mean and standard deviation from `target_sample_rgb` and apply them to the lattice so `LUT(target_pixel) == graded_pixel`.
5. **Font Sizing & Descender Padding**:
   - *Problem*: Default fallback font in Pillow without explicit size caused tiny 10px text, and drawing at `(8, 8)` clipped descenders ('g', 'j', 'p', 'y').
   - *Fix*: Use `load_default(size=font_size)` fallback and offset drawing coordinates with `(10 - bbox[0], 10 - bbox[1])`.
6. **Background Removal Quality Upgrade (U2-Net → BiRefNet)**:
   - *Problem*: U2-Net (2020-era model) produced poor results on hair, fur, glass, and semi-transparent objects — hard binary masks with clipped edges.
   - *Fix*: Upgraded to BiRefNet (Bilateral Reference Network, 2024 state-of-the-art). rembg v2.0.85 supports BiRefNet natively via `new_session(model_name="birefnet-general")`. Added model selector dropdown in sidebar with 4 variants: `birefnet-general` (default, best quality), `birefnet-general-lite` (fast), `birefnet-portrait` (optimized for human hair), `birefnet-massive` (max quality). No new dependencies needed.
   - *Note*: BiRefNet-General is ~970 MB (auto-downloaded on first use). CPU inference ~10-15s/image vs U2-Net's ~0.4s. GPU recommended for production.
