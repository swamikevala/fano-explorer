"""
Annotations Blueprint - Comments and protected regions routes.

Annotations use inline markers in the markdown document.
Markers look like: <!-- @ann:c001 --> for comments, <!-- @ann:p001 --> for protected
The annotation content is stored in annotations.json
"""

from flask import Blueprint, jsonify, request

from ..services import DOC_PATH

bp = Blueprint("annotations", __name__, url_prefix="/api")


@bp.route("/annotations")
def api_annotations():
    """Get all annotations."""
    try:
        from documenter.annotations import AnnotationManager
        from documenter.document import Document

        am = AnnotationManager(DOC_PATH)
        doc = Document(DOC_PATH)
        doc.load()

        # Get markers present in document
        markers_in_doc = set(doc.find_all_markers())

        # Build response with linked/unlinked info
        annotations = {}
        unlinked = []

        for ann_id, ann in am.annotations.items():
            ann_dict = ann.to_dict()
            if ann_id in markers_in_doc:
                ann_dict["linked"] = True
                annotations[ann_id] = ann_dict
            else:
                ann_dict["linked"] = False
                unlinked.append(ann_dict)

        return jsonify({
            "annotations": annotations,
            "unlinked": unlinked,
        })
    except Exception as e:
        return jsonify({"error": str(e), "annotations": {}, "unlinked": []}), 500


@bp.route("/annotations", methods=["POST"])
def api_add_annotation():
    """Add a new annotation (comment or protected)."""
    try:
        data = request.json
        ann_type = data.get("type", "comment")  # "comment" or "protected"
        content = data.get("content", "")
        char_offset = data.get("char_offset", 0)
        text_preview = data.get("text_preview", "")
        # Full selected text for searching in markdown
        search_text = data.get("search_text", text_preview)

        if ann_type == "comment" and not content:
            return jsonify({"error": "Comment content is required"}), 400

        from documenter.annotations import AnnotationManager
        from documenter.document import Document

        # Create annotation (generates ID)
        am = AnnotationManager(DOC_PATH)
        annotation = am.add(ann_type, content, text_preview)

        # Insert marker into document - use search_text to find correct position
        doc = Document(DOC_PATH, enable_versioning=True)
        doc.load()
        doc.insert_marker(annotation.id, char_offset, search_text)
        doc.save(f"Added annotation {annotation.id}")

        return jsonify({
            "success": True,
            "annotation": annotation.to_dict(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/annotations/<ann_id>", methods=["DELETE"])
def api_delete_annotation(ann_id):
    """Delete an annotation and its marker."""
    try:
        from documenter.annotations import AnnotationManager
        from documenter.document import Document

        # Delete from annotations
        am = AnnotationManager(DOC_PATH)
        if not am.delete(ann_id):
            return jsonify({"error": "Annotation not found"}), 404

        # Remove marker from document
        doc = Document(DOC_PATH, enable_versioning=True)
        doc.load()
        doc.remove_marker(ann_id)
        doc.save(f"Removed annotation {ann_id}")

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Legacy endpoints for backwards compatibility
@bp.route("/comments", methods=["POST"])
def api_add_comment():
    """Add a new comment (legacy endpoint)."""
    try:
        data = request.json or {}
        content = data.get("content", "")
        char_offset = data.get("char_offset", 0)
        selected_text = data.get("selected_text", "")
        text_preview = data.get("text_preview", selected_text[:50])
        search_text = data.get("search_text", selected_text)

        if not content:
            return jsonify({"error": "Comment content is required"}), 400

        from documenter.annotations import AnnotationManager
        from documenter.document import Document

        am = AnnotationManager(DOC_PATH)
        annotation = am.add("comment", content, text_preview)

        doc = Document(DOC_PATH, enable_versioning=True)
        doc.load()
        doc.insert_marker(annotation.id, char_offset, search_text)
        doc.save(f"Added comment {annotation.id}")

        return jsonify({"success": True, "comment": annotation.to_dict()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/comments/<comment_id>", methods=["DELETE"])
def api_delete_comment(comment_id):
    """Delete a comment (legacy endpoint)."""
    return api_delete_annotation(comment_id)


@bp.route("/protected", methods=["POST"])
def api_add_protected():
    """Add a new protected region (legacy endpoint)."""
    try:
        data = request.json or {}
        char_offset = data.get("char_offset", 0)
        selected_text = data.get("selected_text", "")
        text_preview = data.get("text_preview", selected_text[:50])
        search_text = data.get("search_text", selected_text)

        from documenter.annotations import AnnotationManager
        from documenter.document import Document

        am = AnnotationManager(DOC_PATH)
        annotation = am.add("protected", "", text_preview)

        doc = Document(DOC_PATH, enable_versioning=True)
        doc.load()
        doc.insert_marker(annotation.id, char_offset, search_text)
        doc.save(f"Added protected region {annotation.id}")

        return jsonify({"success": True, "protected": annotation.to_dict()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/protected/<protected_id>", methods=["DELETE"])
def api_delete_protected(protected_id):
    """Delete a protected region (legacy endpoint)."""
    return api_delete_annotation(protected_id)
