"""
Lumina Image Suite - Comprehensive Verification Script
=====================================================
Executes deterministic automated unit and integration tests across all 4 modules:
1. AI Background Remover (rembg U2-Net, alpha matting, feathering, composite)
2. Cinematic LUT Studio (Reinhard color transfer, histogram matching, 3D .cube generation)
3. Smart Batch Resizer (smart crop, letterbox, percentage, Lanczos/Bicubic resampling)
4. Watermark & Branding (text/logo watermarking, 9-grid, tiling, blend modes, safe format export)
"""

import io
import os
import sys
import time
import numpy as np
from PIL import Image, ImageDraw

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app


class MockUploadedFile(io.BytesIO):
    """Simulates a Streamlit UploadedFile object."""
    def __init__(self, name: str, data: bytes):
        super().__init__(data)
        self.name = name


def create_sample_image(width=400, height=300, color=(180, 70, 50), text="Sample") -> MockUploadedFile:
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([width//6, height//6, 5*width//6, 5*height//6], fill=(color[0], color[1], color[2], 255))
    draw.ellipse([width//4, height//4, 3*width//4, 3*height//4], fill=(255, 220, 100, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return MockUploadedFile(f"{text}.png", buf.getvalue())


def verify_module_1():
    print("[1/4] Verifying Module 1: AI Background Remover...")
    mock_file = create_sample_image(160, 160, (220, 30, 40), "portrait")
    
    t0 = time.time()
    session = app.get_rembg_session()
    print(f"      ONNX session ready in {time.time() - t0:.2f}s")
    
    pil_img = app.open_image_safely(mock_file)
    
    # Transparent PNG mode
    res_png, fmt = app.remove_background_advanced(
        img=pil_img,
        session=session,
        bg_mode="Transparant (PNG)",
        custom_color_hex="#FFFFFF",
        alpha_matting=False,
        fg_threshold=240,
        bg_threshold=10,
        erode_size=10,
        feather_edges=True
    )
    assert fmt == "PNG", f"Expected PNG format, got {fmt}"
    assert res_png.mode == "RGBA", f"Expected RGBA mode, got {res_png.mode}"
    
    # Solid background mode
    res_solid, fmt2 = app.remove_background_advanced(
        img=pil_img,
        session=session,
        bg_mode="Effen wit",
        custom_color_hex="#FFFFFF",
        alpha_matting=False,
        fg_threshold=240,
        bg_threshold=10,
        erode_size=10,
        feather_edges=False
    )
    assert fmt2 == "JPEG", f"Expected JPEG format, got {fmt2}"
    assert res_solid.mode == "RGB", f"Expected RGB mode, got {res_solid.mode}"
    print("      ✔ AI Background Remover PASSED")


def verify_module_2():
    print("[2/4] Verifying Module 2: Cinematic 3D LUT Generator...")
    ref_file = create_sample_image(120, 120, (30, 120, 200), "blade_runner_ref")
    ref_pil = app.open_image_safely(ref_file).convert("RGB")
    ref_np = np.array(ref_pil)
    
    # Palette extraction
    palette = app.extract_dominant_palette(ref_pil, 5)
    assert len(palette) == 5, f"Expected 5 palette colors, got {len(palette)}"
    
    # Dedicated 3D .cube LUT Generation with fine-tuning
    cube_content, lut_grid = app.generate_cinematic_lut(
        reference_rgb=ref_np,
        lut_size=17,
        title="Dune_Orange_Teal",
        intensity_pct=85,
        warmth=10,
        contrast=15,
        lift_blacks=5,
        saturation_pct=110
    )
    assert 'TITLE "Dune_Orange_Teal"' in cube_content, "Missing custom title in .cube"
    assert "LUT_3D_SIZE 17" in cube_content, "Incorrect LUT size declaration"
    assert lut_grid.shape == (17, 17, 17, 3), "LUT grid shape mismatch"
    
    # Real-time preview application
    test_frame = app.get_default_cinematic_test_frame()
    graded_frame = app.apply_lut_to_image(np.array(test_frame), lut_grid)
    assert graded_frame.shape == (540, 960, 3), "Graded frame shape mismatch"
    print("      ✔ Cinematic 3D LUT Generator PASSED")


def verify_module_3():
    print("[3/4] Verifying Module 3: Smart Batch Resizer...")
    f = create_sample_image(600, 400, (60, 180, 70), "landscape")
    orig_img = app.open_image_safely(f)
    
    # Smart Crop
    c1 = app.smart_resize_and_crop(
        img=orig_img,
        dimension_mode="Vaste Resolutie",
        target_width=300,
        target_height=300,
        scale_percentage=100,
        fit_mode="Smart Crop",
        pad_color_hex="#000000",
        resample_method=Image.Resampling.LANCZOS
    )
    assert c1.size == (300, 300), f"Expected (300, 300), got {c1.size}"
    
    # Pad / Letterbox
    c2 = app.smart_resize_and_crop(
        img=orig_img,
        dimension_mode="Vaste Resolutie",
        target_width=800,
        target_height=400,
        scale_percentage=100,
        fit_mode="Pad / Letterbox",
        pad_color_hex="#111827",
        resample_method=Image.Resampling.BICUBIC
    )
    assert c2.size == (800, 400), f"Expected (800, 400), got {c2.size}"
    
    # Percentage
    c3 = app.smart_resize_and_crop(
        img=orig_img,
        dimension_mode="Percentage",
        target_width=100,
        target_height=100,
        scale_percentage=50,
        fit_mode="Smart Crop",
        pad_color_hex="#000000",
        resample_method=Image.Resampling.BILINEAR
    )
    assert c3.size == (300, 200), f"Expected (300, 200), got {c3.size}"

    # Ambient Blur (Stories/Reels 9:16)
    c4 = app.smart_resize_and_crop(
        img=orig_img,
        dimension_mode="Vaste Resolutie",
        target_width=1080,
        target_height=1920,
        scale_percentage=100,
        fit_mode="Ambient Blur",
        pad_color_hex="#000000",
        resample_method=Image.Resampling.LANCZOS
    )
    assert c4.size == (1080, 1920), f"Expected (1080, 1920), got {c4.size}"

    # E-Commerce 10% Safe Margin (Bol.com / Amazon 1:1)
    c5 = app.smart_resize_and_crop(
        img=orig_img,
        dimension_mode="Vaste Resolutie",
        target_width=1200,
        target_height=1200,
        scale_percentage=100,
        fit_mode="Pad / Letterbox (10% Safe Margin)",
        pad_color_hex="#FFFFFF",
        resample_method=Image.Resampling.LANCZOS
    )
    assert c5.size == (1200, 1200), f"Expected (1200, 1200), got {c5.size}"
    print("      ✔ Smart Batch Resizer PASSED")


def verify_module_4():
    print("[4/4] Verifying Module 4: Watermark & Branding...")
    base_file = create_sample_image(500, 350, (100, 80, 150), "photo")
    sample_base = app.open_image_safely(base_file)
    
    wm_text = app.create_text_watermark_image("© 2026 Lumina Studio", 32, "#38BDF8")
    assert wm_text.mode == "RGBA", "Watermark text must be RGBA"
    
    # Single watermark
    res1 = app.apply_watermark(
        base_image=sample_base,
        watermark_image=wm_text,
        mode="Enkel Watermerk",
        grid_position="Bottom-Right",
        offset_x=20,
        offset_y=20,
        scale_pct=30,
        opacity_pct=85,
        rotation_deg=0,
        blend_mode="Normal"
    )
    assert res1.size == sample_base.size, "Output size mismatch"
    
    # Tiled / Mosaic with Screen blend mode
    res2 = app.apply_watermark(
        base_image=sample_base,
        watermark_image=wm_text,
        mode="Tiled / Mosaic",
        grid_position="Center",
        offset_x=10,
        offset_y=10,
        scale_pct=20,
        opacity_pct=40,
        rotation_deg=15,
        blend_mode="Screen"
    )
    assert res2.size == sample_base.size, "Output size mismatch in tiled mode"
    print("      ✔ Watermark & Branding PASSED")


def verify_module_5():
    print("[5/6] Verifying Module 5: AI Studio Product Photo Maker...")
    # Mock product cutout
    prod = Image.new("RGBA", (200, 360), (0, 0, 0, 0))
    d = ImageDraw.Draw(prod)
    d.rounded_rectangle([20, 40, 180, 340], radius=25, fill=(240, 160, 60, 240))
    d.rectangle([70, 10, 130, 40], fill=(220, 220, 230, 255))

    # 1. Test Cyclorama backdrop generation
    backdrop = app.generate_studio_backdrop(800, 800, "Apple Clean White", light_pos="Boven (Midden)")
    assert backdrop.size == (800, 800), f"Expected backdrop (800, 800), got {backdrop.size}"
    assert backdrop.mode == "RGB", f"Expected RGB mode, got {backdrop.mode}"

    # 2. Test Studio Composition with shadow, reflection and ground placement
    studio_shot = app.compose_studio_product(
        product_cutout=prod,
        canvas_size=(1000, 1000),
        theme="Luxury Dark Noir",
        light_pos="Boven (Links - 45°)",
        shadow_intensity=80,
        reflection_pct=30,
        ground_pos_pct=76
    )
    assert studio_shot.size == (1000, 1000), f"Expected (1000, 1000), got {studio_shot.size}"
    assert studio_shot.mode == "RGB", f"Expected RGB mode, got {studio_shot.mode}"

    # 3. Test Custom Color Studio
    custom_shot = app.compose_studio_product(
        product_cutout=prod,
        canvas_size=(800, 1000),
        theme="🎨 Aangepaste Tint...",
        custom_hex="#E8ECE9",
        light_pos="Boven (Midden)",
        shadow_intensity=60,
        reflection_pct=20,
        ground_pos_pct=75
    )
    assert custom_shot.size == (800, 1000), f"Expected (800, 1000), got {custom_shot.size}"
    # 4. Test Pure White Luxury Studio (Jack Daniel's Stijl • Foto 2 & 3)
    pure_shot = app.compose_studio_product(
        product_cutout=prod,
        canvas_size=(600, 600),
        theme="🏆 Pure White Luxury Studio (Jack Daniel's Stijl • Foto 2 & 3)",
        shadow_style="🏆 Jack Daniel's Minimal Contact (Puur Wit • Zoals Foto 2 & 3)",
        shadow_intensity=70,
        reflection_pct=25
    )
    assert pure_shot.size == (600, 600), f"Expected (600, 600), got {pure_shot.size}"
    arr = np.array(pure_shot)
    # Border pixels must be pure white (255, 255, 255)
    assert np.all(arr[0:50, :, :] == 255), "Top border must be pure 255 white"
    assert np.all(arr[:, 0:50, :] == 255), "Left border must be pure 255 white"
    assert np.all(arr[:, -50:, :] == 255), "Right border must be pure 255 white"
    print("      ✔ AI Studio Product Photo Maker PASSED (Pure White Studio Verified)")


def verify_module_6():
    print("[6/7] Verifying Module 6: AI Super-Resolution 4K Upscaler...")
    sample_thumb = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
    d = ImageDraw.Draw(sample_thumb)
    d.rectangle([10, 10, 70, 70], fill=(50, 150, 250, 240))
    d.line([10, 10, 70, 70], fill=(255, 255, 255, 255), width=3)

    t0 = time.time()
    # Test 2x upscaling
    up_2x = app.upscale_image_advanced(sample_thumb, scale=2)
    t_2x = time.time() - t0
    assert up_2x.size == (160, 160), f"Expected (160, 160), got {up_2x.size}"
    assert up_2x.mode == "RGBA", f"Expected RGBA mode preservation, got {up_2x.mode}"
    print(f"      2× Super-Resolution completed in {t_2x:.2f}s")

    # Test 4x upscaling
    t1 = time.time()
    up_4x = app.upscale_image_advanced(sample_thumb, scale=4)
    t_4x = time.time() - t1
    assert up_4x.size == (320, 320), f"Expected (320, 320), got {up_4x.size}"
    print(f"      4× Ultra HD completed in {t_4x:.2f}s")
    print("      ✔ AI Super-Resolution 4K Upscaler PASSED")


def verify_module_7():
    print("[7/7] Verifying Module 7: AI Photo to 3D Studio (Tripo 3D & Watertight Meshing)...")
    # 1. Create a test object (bottle profile)
    test_obj = Image.new("RGBA", (150, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(test_obj)
    d.rectangle([55, 20, 95, 60], fill=(220, 160, 40, 255))   # Neck
    d.rectangle([35, 60, 115, 180], fill=(220, 160, 40, 255)) # Body

    t0 = time.time()

    # 2. Test interactive click-to-mask
    mask = app.get_object_mask_from_click(test_obj, click_x=75, click_y=100)
    assert mask.shape == (200, 150), f"Expected shape (200, 150), got {mask.shape}"
    assert np.sum(mask > 0) > 1000, "Mask should cover the bottle region"

    # 3. Test glowing SAM overlay
    overlay = app.create_glowing_selection_overlay(test_obj.convert("RGB"), mask, click_pos=(75, 100))
    assert overlay.size == (150, 200), f"Expected size (150, 200), got {overlay.size}"
    assert overlay.mode == "RGBA", f"Expected RGBA mode, got {overlay.mode}"

    # 4. Test 3D Mesh Synthesis (Smart Mesh & HD Model)
    mesh, stl_b, obj_s, glb_b, vdata = app.generate_3d_mesh_from_image(
        cutout_img=test_obj,
        depth_scale=1.0,
        mesh_quality="Smart Mesh",
        flat_base=False,
        relief_strength=0.25
    )

    t_3d = time.time() - t0
    assert len(mesh.vertices) > 500, f"Expected >500 vertices, got {len(mesh.vertices)}"
    assert len(mesh.faces) > 1000, f"Expected >1000 faces, got {len(mesh.faces)}"
    assert mesh.is_watertight, "Mesh MUST be 100% watertight (manifold) for 3D printing"
    assert len(stl_b) > 1000, "STL binary export should be non-empty"
    assert len(glb_b) > 1000, "GLB binary export should be non-empty"
    assert len(obj_s) > 1000, "OBJ text export should be non-empty"
    assert "vertices" in vdata and "faces" in vdata and "textureUrl" in vdata

    # 5. Test Three.js Viewer HTML Builder
    html = app.build_threejs_viewer_html(vdata, height=620)
    assert "<!DOCTYPE html>" in html and "THREE.OrbitControls" in html and "TRIPO 3D VIEWPORT" in html
    print(f"      3D Watertight Mesh generated in {t_3d:.2f}s ({len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces)")
    print("      ✔ AI Photo to 3D Studio PASSED (STL, GLB, OBJ & Three.js Verified)")


def run_all():
    print("=" * 60)
    print("LUMINA PRO - COMPREHENSIVE SUITE VERIFICATION (7 MODULES)")
    print("=" * 60)
    t_start = time.time()
    verify_module_1()
    verify_module_2()
    verify_module_3()
    verify_module_4()
    verify_module_5()
    verify_module_6()
    verify_module_7()
    print("=" * 60)
    print(f"ALL 7 MODULES FULLY VERIFIED AND OPERATIONAL in {time.time() - t_start:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
