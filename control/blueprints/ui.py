"""
UI Blueprint - Dashboard and document viewer routes.
"""

from flask import Blueprint, render_template

from ..services import load_config, DOC_PATH

bp = Blueprint("ui", __name__)


@bp.route("/")
def index():
    """Main dashboard."""
    config = load_config()
    return render_template("index.html", config=config)


@bp.route("/document")
def view_document():
    """View the document in browser."""
    if not DOC_PATH.exists():
        return "<h1>Document not found</h1><p>The document hasn't been created yet. Start the documenter to begin.</p>"

    try:
        import markdown
        content = DOC_PATH.read_text(encoding="utf-8")
        html_content = markdown.markdown(content, extensions=['tables', 'fenced_code', 'toc'])
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fano Document</title>
            <style>
                body {{ font-family: Georgia, serif; max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
                h1, h2, h3 {{ color: #333; }}
                h1 {{ border-bottom: 2px solid #8b5cf6; padding-bottom: 10px; }}
                h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 30px; }}
                code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
                pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                blockquote {{ border-left: 4px solid #8b5cf6; margin: 0; padding-left: 20px; color: #666; }}
                a {{ color: #8b5cf6; }}
                .nav {{ background: #333; color: white; padding: 10px 20px; margin: -20px -20px 20px -20px; }}
                .nav a {{ color: #8b5cf6; margin-right: 20px; }}
            </style>
        </head>
        <body>
            <div class="nav">
                <a href="/">← Dashboard</a>
                <span>Fano Living Document</span>
            </div>
            {html_content}
        </body>
        </html>
        """
    except ImportError:
        # Fallback if markdown not installed
        content = DOC_PATH.read_text(encoding="utf-8")
        return f"<pre>{content}</pre>"
    except Exception as e:
        return f"<h1>Error</h1><p>{e}</p>"
