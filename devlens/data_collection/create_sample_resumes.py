import os
import docx
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_pdf(file_path: str, lines: list):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    c = canvas.Canvas(file_path, pagesize=letter)
    y = 750
    for line in lines:
        c.drawString(50, y, line)
        y -= 25
    c.save()


def create_docx(file_path: str, lines: list):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    doc = docx.Document()
    for line in lines:
        doc.add_paragraph(line)
    doc.save(file_path)


def generate_all_samples(output_dir: str = "data/sample_resumes"):
    # 1. Direct URL PDF Resume
    create_pdf(
        os.path.join(output_dir, "resume_direct_url.pdf"),
        [
            "Linus Torvalds - Principal Kernel Architect",
            "Email: torvalds@kernel.org",
            "GitHub: https://github.com/torvalds",
            "Summary: Creator of Linux and Git.",
            "Skills: C, Assembly, Systems Programming, Git Architecture"
        ]
    )

    # 2. Labeled Handle DOCX Resume
    create_docx(
        os.path.join(output_dir, "resume_labeled.docx"),
        [
            "Linus Torvalds",
            "Location: Portland, OR",
            "GitHub Handle: torvalds",
            "Experience: Linux Foundation - Fellow",
            "Projects: Linux Kernel, Git, Subsurface"
        ]
    )

    # 3. Inferred Fallback PDF Resume (No direct link/label)
    create_pdf(
        os.path.join(output_dir, "resume_inferred.pdf"),
        [
            "Linus Torvalds",
            "Email: torvalds@kernel.org",
            "Location: Portland, OR",
            "Role: Chief Linux Architect",
            "Summary: Experienced software engineer focusing on OS kernels."
        ]
    )

    # 4. No GitHub Match DOCX Resume
    create_docx(
        os.path.join(output_dir, "resume_no_match.docx"),
        [
            "Jane NonexistentUserXYZ123",
            "Email: jane.nonexistent12389@fake-domain-not-real.com",
            "Role: Junior Developer",
            "Summary: Seeking software development roles."
        ]
    )

    print(f"Sample resumes created successfully in '{output_dir}'.")


if __name__ == "__main__":
    generate_all_samples()
