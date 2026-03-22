"""Login, register, logout, JWT token & refresh endpoints."""
from __future__ import annotations

from datetime import timedelta

from flask import Blueprint, jsonify, redirect, render_template, request, url_for
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    get_jwt_identity,
    jwt_required,
    set_access_cookies,
    set_refresh_cookies,
    unset_jwt_cookies,
)

from auth.db import create_user, verify_user

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


def _safe_next_url(next_param: str | None) -> str | None:
    if not next_param or not next_param.startswith("/"):
        return None
    if next_param.startswith("//"):
        return None
    return next_param


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template(
            "login.html",
            error=None,
            next_url=request.args.get("next") or "",
        )

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    if not username or not password:
        return (
            render_template(
                "login.html",
                error="Username and password are required.",
                next_url=request.form.get("next") or "",
            ),
            400,
        )

    if not verify_user(username, password):
        return (
            render_template(
                "login.html",
                error="Invalid username or password.",
                next_url=request.form.get("next") or "",
            ),
            401,
        )

    access = create_access_token(identity=username, fresh=True)
    refresh = create_refresh_token(identity=username)
    nxt = _safe_next_url(request.args.get("next") or request.form.get("next"))
    resp = redirect(nxt or url_for("index"))
    set_access_cookies(resp, access)
    set_refresh_cookies(resp, refresh)
    return resp


def _issue_token_response():
    """JSON login: returns access + refresh JWT (for API clients)."""
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username or not password:
        return jsonify({"ok": False, "error": "username and password required"}), 400
    if not verify_user(username, password):
        return jsonify({"ok": False, "error": "invalid credentials"}), 401

    access = create_access_token(identity=username, fresh=True)
    refresh = create_refresh_token(identity=username)
    return jsonify(
        {
            "ok": True,
            "access_token": access,
            "refresh_token": refresh,
            "token_type": "Bearer",
            "expires_in": int(timedelta(hours=1).total_seconds()),
        }
    )


@auth_bp.route("/login/json", methods=["POST"])
def login_json():
    return _issue_token_response()


@auth_bp.route("/token", methods=["POST"])
def token():
    """Alias for JSON login (Bearer tokens for API clients)."""
    return _issue_token_response()


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html", error=None)

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    if not username or not password:
        return (
            render_template(
                "register.html", error="Username and password are required."
            ),
            400,
        )

    ok, msg = create_user(username, password)
    if not ok:
        return render_template("register.html", error=msg), 400

    access = create_access_token(identity=username, fresh=True)
    refresh = create_refresh_token(identity=username)
    resp = redirect(url_for("index"))
    set_access_cookies(resp, access)
    set_refresh_cookies(resp, refresh)
    return resp


@auth_bp.route("/logout", methods=["POST", "GET"])
def logout():
    resp = redirect(url_for("auth.login"))
    unset_jwt_cookies(resp)
    return resp


@auth_bp.route("/refresh", methods=["POST"])
@jwt_required(refresh=True)
def refresh():
    """Exchange refresh JWT for new access JWT (Authorization: Bearer <refresh>)."""
    identity = get_jwt_identity()
    access = create_access_token(identity=identity, fresh=False)
    out = jsonify({"ok": True, "access_token": access, "token_type": "Bearer"})
    set_access_cookies(out, access)
    return out


@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def me():
    return jsonify({"username": get_jwt_identity()})
