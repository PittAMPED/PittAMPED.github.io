# === Imports ===
from pathlib import Path
import markdown2
import os
import re
import shutil

# =============================================================================
# CONFIGURATION — edit this section when you rename folders
# =============================================================================

SECTION_MAP = {
    "nanocrystalline": "nanocrystalline",   # nanocrystalline_directory.md → nanocrystalline/
    "ferrites":        "ferrites",           # ferrites_directory.md        → ferrites/
    # Add new sections here as you create them:
    # "my_new_section": "my_new_folder",
}

# Folders inside material_database/ to always ignore
SKIP_FOLDERS = {"Template Alloy", "database"}

# =============================================================================

# === Helpers ===
def slugify(text):
    return re.sub(r"[^\w\-]", "", text.strip().replace(" ", ""))

# === Configuration Paths ===
base = Path(__file__).resolve().parent

input_dir    = base / "material_database"
database_dir = input_dir / "database"
output_root  = base / "public_database"
template_dir = output_root


# === Auto-detect all sections ===
def detect_sections():
    sections = []

    if not database_dir.exists():
        print(f"⚠ database/ folder not found at {database_dir}")
        return sections

    for section_key, folder_name in SECTION_MAP.items():
        md_file = database_dir / f"{section_key}_directory.md"
        if not md_file.exists():
            print(f"⚠ Directory file not found: {md_file.name} — skipping '{section_key}'.")
            continue

        material_root = input_dir / folder_name
        if not material_root.exists() or not material_root.is_dir():
            print(f"⚠ Material folder not found: material_database/{folder_name}/ — skipping '{section_key}'.")
            continue

        section_output_dir = output_root / f"{section_key}_public"
        section_output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-create nc_template.html if missing
        template_html = section_output_dir / "nc_template.html"
        master_template = template_dir / "material_template.html"
        if not template_html.exists() and master_template.exists():
            shutil.copy2(master_template, template_html)
            print(f"  📋 Created missing nc_template.html for '{section_key}'")

        sections.append({
            "name":          section_key,
            "directory_md":  md_file,
            "material_root": material_root,
            "output_dir":    section_output_dir,
            "index_html":    section_output_dir / f"{section_key}_index.html",
            "template_html": template_html,
            "material_tmpl": master_template,
        })
        print(f"📂 Section: '{section_key}' → material_database/{folder_name}/ → {section_output_dir.name}/")

    return sections


# === Per-section maps ===
wiki_link_map = {}
html_file_map = {}
image_map     = {}


def _build_maps(section):
    global wiki_link_map, html_file_map, image_map
    wiki_link_map = {}
    html_file_map = {}
    image_map     = {}

    for md_file in section["material_root"].rglob("*.md"):
        if md_file.name.startswith(("_", ".")):
            continue
        key      = slugify(md_file.stem)
        rel_path = md_file.relative_to(section["material_root"]).with_suffix(".html")
        wiki_link_map[key]     = rel_path
        html_file_map[md_file] = rel_path


def _build_image_map(section):
    """Scan markdown files and locate all referenced images.
    Searches ALL subfolders recursively so any folder naming works:
    1Images, 1 Images, 30Images, 3 Images, etc.
    """
    material_root = section["material_root"]
    image_exts    = [".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"]

    print(f"🔍 [{section['name']}] Building image database...")

    # Build a flat map of ALL images anywhere under material_root
    # filename → full path (last one wins if duplicates)
    all_images = {}
    for img_path in material_root.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in image_exts:
            all_images[img_path.name] = img_path
            # Also index without extension for obsidian-style refs
            all_images[img_path.stem] = img_path

    for md_file in material_root.rglob("*.md"):
        if md_file.name.startswith(("_", ".")):
            continue
        try:
            content    = md_file.read_text(encoding="utf-8")
            image_refs = re.findall(r"!\[\[([^\]]+)\]\]", content)

            for image_ref in image_refs:
                if image_ref in image_map:
                    continue

                # First try exact filename match in flat map
                found_image = all_images.get(image_ref)

                # Then try without extension
                if not found_image:
                    found_image = all_images.get(image_ref.rsplit(".", 1)[0] if "." in image_ref else image_ref)

                # Fallback: search all subdirs of the alloy's numbered folder
                if not found_image:
                    search_root = md_file.parent
                    for loc in [search_root] + list(search_root.rglob("*")):
                        if not hasattr(loc, 'is_dir') or not loc.is_dir():
                            continue
                        # Try exact name
                        test = loc / image_ref
                        if test.exists():
                            found_image = test
                            break
                        # Try with each extension
                        if "." not in image_ref:
                            for ext in image_exts:
                                test = loc / f"{image_ref}{ext}"
                                if test.exists():
                                    found_image = test
                                    break
                        if found_image:
                            break

                if found_image:
                    image_map[image_ref] = found_image.relative_to(material_root)
                    print(f"  ✅ Found image: {image_ref}")
                else:
                    image_map[image_ref] = None
                    print(f"  ❌ Image not found: {image_ref}")

        except Exception as e:
            print(f"  ❌ Error reading {md_file}: {e}")

    found   = sum(1 for v in image_map.values() if v is not None)
    missing = sum(1 for v in image_map.values() if v is None)
    print(f"  📊 Images: {found} found, {missing} missing")


# === Content Processing ===
def apply_publish_stop(content):
    if "<!-- PUBLISH STOP -->" in content:
        content = content.split("<!-- PUBLISH STOP -->", 1)[0].strip()
    return content


def process_markdown(content, section, current_md_path=None, is_index_page=False):
    material_output_root = section["output_dir"] / "alloys"

    def replace_wiki(m):
        raw_target  = m.group(1).strip()
        label       = m.group(3) or raw_target
        target_slug = slugify(raw_target)
        target_rel  = wiki_link_map.get(target_slug)

        if not target_rel:
            return f'<a href="#" style="color: red;">Link not found: {label}</a>'

        if is_index_page:
            href = f"alloys/{target_rel.as_posix()}"
        elif current_md_path and current_md_path in html_file_map:
            current_rel = html_file_map[current_md_path]
            href = os.path.relpath(
                material_output_root / target_rel,
                start=(material_output_root / current_rel).parent
            ).replace("\\", "/")
        else:
            href = target_rel.as_posix()

        return f'<a href="{href}">{label}</a>'

    def replace_image(m):
        name = m.group(1).strip()
        if name not in image_map or image_map[name] is None:
            return f"<div><strong>Missing Image:</strong> {name}</div>"

        rel_path = image_map[name]
        if is_index_page:
            src = f"alloys/{rel_path.as_posix()}"
        elif current_md_path and current_md_path in html_file_map:
            current_html = material_output_root / html_file_map[current_md_path]
            src = os.path.relpath(
                material_output_root / rel_path,
                start=current_html.parent
            ).replace("\\", "/")
        else:
            src = rel_path.as_posix()

        return f'<img src="{src}" alt="{name}" style="max-width:100%;">'

    content = re.sub(r"\[\[([^\[\]|]+)(\|([^\]]+))?\]\]", replace_wiki, content)
    content = re.sub(r"!\[\[([^\]]+)\]\]", replace_image, content)
    return markdown2.markdown(
        content,
        extras=["fenced-code-blocks", "tables", "footnotes", "break-on-newline"]
    )


# === Copy Images ===
def _copy_images(section):
    material_output_root = section["output_dir"] / "alloys"
    print(f"\n📁 [{section['name']}] Copying images...")
    copied = 0
    for name, rel in image_map.items():
        if rel is None:
            continue
        src = section["material_root"] / rel
        dst = material_output_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)
            copied += 1
            print(f"  ✅ Copied: {rel}")
    print(f"  📊 Images copied: {copied}")


# === Generate Index Page ===
def _generate_index_page(section):
    print(f"\n📄 [{section['name']}] Generating index page...")

    template_path = section["template_html"]
    if not template_path.exists():
        print(f"  ⚠ Template not found: {template_path} — skipping.")
        return

    md_file = section["directory_md"]
    content = apply_publish_stop(md_file.read_text(encoding="utf-8"))
    html    = process_markdown(content, section, current_md_path=md_file, is_index_page=True)

    template = template_path.read_text(encoding="utf-8")
    section["index_html"].write_text(
        template.replace("<!-- CONTENT GOES HERE -->", html),
        encoding="utf-8"
    )
    print(f"  ✅ Created index: {section['index_html'].relative_to(base)}")


# === Generate Individual Material Pages ===
def _generate_material_pages(section):
    material_root        = section["material_root"]
    material_output_root = section["output_dir"] / "alloys"

    print(f"\n📄 [{section['name']}] Generating material pages...")

    template_path = section["material_tmpl"]
    if not template_path.exists():
        print(f"  ⚠ material_template.html not found — skipping.")
        return
    template       = template_path.read_text(encoding="utf-8")
    directory_stem = section["directory_md"].stem.lower()

    for md_file in sorted(material_root.rglob("*.md")):
        if md_file.name.startswith(("_", ".")):
            continue
        if md_file.stem.lower() == directory_stem:
            continue

        slug     = slugify(md_file.stem)
        out_file = material_output_root / md_file.relative_to(material_root).parent / f"{slug}.html"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        content           = md_file.read_text(encoding="utf-8")
        processed_content = apply_publish_stop(content)
        html              = process_markdown(processed_content, section, current_md_path=md_file, is_index_page=False)
        title             = processed_content.strip().split("\n", 1)[0].lstrip("#").strip() or slug.replace("_", " ").title()

        out_file.write_text(
            template
            .replace("<!-- CONTENT GOES HERE -->", html)
            .replace("<!-- TITLE GOES HERE -->", title),
            encoding="utf-8"
        )
        print(f"  ✅ Created: {out_file.relative_to(base)}")


# =============================================================================
# Functions called by GUI
# =============================================================================

def build_image_map():
    for section in detect_sections():
        _build_maps(section)
        _build_image_map(section)

def copy_images():
    for section in detect_sections():
        _build_maps(section)
        _build_image_map(section)
        _copy_images(section)

def generate_index_pages():
    for section in detect_sections():
        _build_maps(section)
        _generate_index_page(section)

def generate_material_pages():
    for section in detect_sections():
        _build_maps(section)
        _build_image_map(section)
        _generate_material_pages(section)


# =============================================================================
# Main
# =============================================================================

def run_all():
    print("🔍 Loading sections from SECTION_MAP...\n")
    sections = detect_sections()

    if not sections:
        print("❌ No sections found. Check SECTION_MAP at the top of this file.")
        return

    print(f"\n✅ Found {len(sections)} section(s): {[s['name'] for s in sections]}\n")

    for section in sections:
        print(f"\n{'='*50}")
        print(f"  Processing: {section['name']}")
        print(f"{'='*50}")
        _build_maps(section)
        _build_image_map(section)
        _copy_images(section)
        _generate_index_page(section)
        _generate_material_pages(section)

    print("\n🎉 All sections processed!")


if __name__ == "__main__":
    run_all()
