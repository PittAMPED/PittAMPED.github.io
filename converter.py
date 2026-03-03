# === Imports ===
from pathlib import Path
import markdown2
import os
import re
import shutil

# === Helpers ===
def slugify(text):
    return re.sub(r"[^\w\-]", "", text.strip().replace(" ", ""))

# === Configuration Paths ===
# Use __file__ so paths are always relative to THIS script,
# regardless of how or from where the script is launched
base = Path(__file__).resolve().parent

input_dir    = base / "material_database"
database_dir = input_dir / "database"   # contains all _directory.md files
output_root  = base / "public_database"
template_dir = output_root              # material_template.html lives here

# Folders to always skip inside material_database/
SKIP_FOLDERS = {"Template Alloy", "database"}


# === Auto-detect all sections from database/ folder ===
def detect_sections():
    """
    Scans database/ for .md files. Each one maps to a material folder.
    e.g. lauren_directory.md → material_database/lauren/
    Returns a list of section dicts.
    """
    sections = []

    if not database_dir.exists():
        print(f"⚠ database/ folder not found at {database_dir}")
        return sections

    for md_file in sorted(database_dir.glob("*.md")):
        if md_file.name.startswith(("_", ".")):
            continue

        # Strip _directory suffix to get the section name
        stem = md_file.stem
        name = re.sub(r"[_\-]directory$", "", stem, flags=re.IGNORECASE)

        # Find matching folder in material_database/ (case-insensitive)
        material_root = None
        for folder in input_dir.iterdir():
            if not folder.is_dir():
                continue
            if folder.name in SKIP_FOLDERS:
                continue
            if folder.name.lower() == name.lower():
                material_root = folder
                break

        if material_root is None:
            print(f"⚠ No matching material folder found for '{md_file.name}' (looked for '{name}') — skipping.")
            continue

        # Output paths for this section
        section_output_dir = output_root / f"{name}_public"
        section_output_dir.mkdir(parents=True, exist_ok=True)

        sections.append({
            "name":          name,
            "directory_md":  md_file,
            "material_root": material_root,
            "output_dir":    section_output_dir,
            "index_html":    section_output_dir / f"{name}_index.html",
            "template_html": section_output_dir / "nc_template.html",
            "material_tmpl": template_dir / "material_template.html",
        })
        print(f"📂 Detected section: '{name}' → {material_root.name}/ → {section_output_dir.name}/")

    return sections


# === Per-section maps (rebuilt for each section) ===
wiki_link_map = {}
html_file_map = {}
image_map     = {}


def build_maps_for_section(section):
    """Rebuild wiki/image maps for a given section."""
    global wiki_link_map, html_file_map, image_map
    wiki_link_map = {}
    html_file_map = {}
    image_map     = {}

    material_root        = section["material_root"]

    for md_file in material_root.rglob("*.md"):
        if md_file.name.startswith(("_", ".")):
            continue
        key      = slugify(md_file.stem)
        rel_path = md_file.relative_to(material_root).with_suffix(".html")
        wiki_link_map[key] = rel_path
        html_file_map[md_file] = rel_path


# === Step 2: Build Image Map ===
def build_image_map(section):
    material_root = section["material_root"]

    print(f"🔍 [{section['name']}] Building image database...")
    image_exts = [".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"]

    for md_file in material_root.rglob("*.md"):
        if md_file.name.startswith(("_", ".")):
            continue
        try:
            content    = md_file.read_text(encoding="utf-8")
            image_refs = re.findall(r"!\[\[([^\]]+)\]\]", content)

            for image_ref in image_refs:
                if image_ref in image_map:
                    continue

                search_locations = [
                    md_file.parent,
                    md_file.parent / "images",
                    md_file.parent / "Images",
                    md_file.parent / "1 images",
                    md_file.parent / "1 Images",
                ]
                parent_dir = md_file.parent
                while parent_dir != material_root and parent_dir.parent != parent_dir:
                    search_locations.extend([
                        parent_dir / "images",
                        parent_dir / "Images",
                        parent_dir / "1 images",
                        parent_dir / "1 Images",
                    ])
                    parent_dir = parent_dir.parent

                found_image = None
                if not any(image_ref.lower().endswith(ext) for ext in image_exts):
                    for loc in search_locations:
                        if not loc.exists():
                            continue
                        for ext in image_exts:
                            test = loc / f"{image_ref}{ext}"
                            if test.exists():
                                found_image = test
                                break
                        if found_image:
                            break
                else:
                    for loc in search_locations:
                        test = loc / image_ref
                        if test.exists():
                            found_image = test
                            break

                if found_image:
                    rel_path             = found_image.relative_to(material_root)
                    image_map[image_ref] = rel_path
                    print(f"  ✅ Found image: {image_ref}")
                else:
                    image_map[image_ref] = None
                    print(f"  ❌ Image not found: {image_ref}")

        except Exception as e:
            print(f"  ❌ Error reading {md_file}: {e}")

    found   = sum(1 for v in image_map.values() if v is not None)
    missing = sum(1 for v in image_map.values() if v is None)
    print(f"  📊 Images: {found} found, {missing} missing")


# === Step 3: Content Processing ===
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


# === Step 4: Copy Images ===
def copy_images(section):
    material_root        = section["material_root"]
    material_output_root = section["output_dir"] / "alloys"

    print(f"\n📁 [{section['name']}] Copying images...")
    copied = 0
    for name, rel in image_map.items():
        if rel is None:
            continue
        src = material_root / rel
        dst = material_output_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)
            copied += 1
            print(f"  ✅ Copied: {rel}")
    print(f"  📊 Images copied: {copied}")


# === Step 5: Generate Index Page ===
def generate_index_page(section):
    print(f"\n📄 [{section['name']}] Generating index page...")

    template_path = section["template_html"]
    if not template_path.exists():
        print(f"  ⚠ Template not found: {template_path} — skipping index page.")
        return

    md_file = section["directory_md"]
    content = md_file.read_text(encoding="utf-8")
    content = apply_publish_stop(content)
    html    = process_markdown(content, section, current_md_path=md_file, is_index_page=True)

    template = template_path.read_text(encoding="utf-8")
    section["index_html"].write_text(
        template.replace("<!-- CONTENT GOES HERE -->", html),
        encoding="utf-8"
    )
    print(f"  ✅ Created index: {section['index_html'].relative_to(base)}")


# === Step 6: Generate Individual Material Pages ===
def generate_material_pages_for_section(section):
    material_root        = section["material_root"]
    material_output_root = section["output_dir"] / "alloys"

    print(f"\n📄 [{section['name']}] Generating material pages...")

    template_path = section["material_tmpl"]
    if not template_path.exists():
        print(f"  ⚠ material_template.html not found at {template_path} — skipping.")
        return
    template = template_path.read_text(encoding="utf-8")

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

        final_html = (
            template
            .replace("<!-- CONTENT GOES HERE -->", html)
            .replace("<!-- TITLE GOES HERE -->", title)
        )
        out_file.write_text(final_html, encoding="utf-8")
        print(f"  ✅ Created: {out_file.relative_to(base)}")


# === Functions exposed to GUI ===
# The GUI calls these by name, so they must exist at module level.

def build_image_map():
    for section in detect_sections():
        build_maps_for_section(section)
        build_image_map(section)

def copy_images_all():
    for section in detect_sections():
        build_maps_for_section(section)
        build_image_map(section)
        copy_images(section)

def generate_index_pages():
    for section in detect_sections():
        build_maps_for_section(section)
        generate_index_page(section)

def generate_material_pages():
    for section in detect_sections():
        build_maps_for_section(section)
        build_image_map(section)
        generate_material_pages_for_section(section)


# === Main Execution ===
def run_all():
    print("🔍 Detecting sections from database/ folder...\n")
    sections = detect_sections()

    if not sections:
        print("❌ No sections found. Make sure database/ contains .md files with matching folders in material_database/.")
        return

    print(f"\n✅ Found {len(sections)} section(s): {[s['name'] for s in sections]}\n")

    for section in sections:
        print(f"\n{'='*50}")
        print(f"  Processing: {section['name']}")
        print(f"{'='*50}")
        build_maps_for_section(section)
        build_image_map(section)
        copy_images(section)
        generate_index_page(section)
        generate_material_pages_for_section(section)

    print("\n🎉 All sections processed!")


if __name__ == "__main__":
    run_all()
