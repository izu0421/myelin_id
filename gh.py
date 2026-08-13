"""GitHub Contents-API helpers: commit text (CSV) and binary (mask) files.

Token is read from Streamlit secrets (never hard-coded). Mirrors the pattern in
the qwen annotation webapp: GET the file for its sha, then PUT base64 content.
"""
from __future__ import annotations
import base64
import json
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import streamlit as st


def get_token() -> str:
    """Read a GitHub PAT from st.secrets['github']['token'] or GITHUB_TOKEN."""
    try:
        sec = st.secrets.get("github", {})
        if isinstance(sec, dict) and str(sec.get("token", "")).strip():
            return str(sec["token"]).strip()
    except Exception:
        pass
    try:
        t = str(st.secrets.get("GITHUB_TOKEN", "")).strip()
        if t:
            return t
    except Exception:
        pass
    return ""


def _request(url: str, token: str, method: str = "GET", payload: dict | None = None):
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            return resp.getcode(), (json.loads(body) if body else {})
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body) if body else {}
        except Exception:
            parsed = {"message": body}
        return e.code, parsed


def commit_bytes(content: bytes, repo_spec: str, branch: str, file_path: str,
                 token: str, message: str) -> tuple[bool, str]:
    """Create/update a file in the repo with raw bytes via the Contents API."""
    if "/" not in repo_spec:
        return False, "Repo must be owner/name."
    owner, repo = (s.strip() for s in repo_spec.split("/", 1))
    branch = (branch or "main").strip()
    file_path = file_path.strip().strip("/")
    token = token.strip()
    if not owner or not repo or not file_path:
        return False, "owner, repo, and path are required."
    if not token:
        return False, "GitHub token is required (set it in secrets)."

    enc_path = quote(file_path, safe="/")
    get_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{enc_path}?ref={quote(branch)}"
    status, data = _request(get_url, token, "GET")
    sha = None
    if status == 200:
        sha = str(data.get("sha", "")).strip() or None
        # skip if identical
        try:
            if base64.b64decode(str(data.get("content", ""))) == content:
                return True, f"unchanged: {file_path}"
        except Exception:
            pass
    elif status != 404:
        return False, f"read failed: {data.get('message', status)}"

    payload = {
        "message": message.strip() or f"Update {file_path}",
        "content": base64.b64encode(content).decode("ascii"),
        "branch": branch,
    }
    if sha is not None:
        payload["sha"] = sha
    put_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{enc_path}"
    pstatus, pdata = _request(put_url, token, "PUT", payload)
    if pstatus in (200, 201):
        new_sha = pdata.get("content", {}).get("sha", "")[:7]
        return True, f"committed {file_path} ({new_sha})"
    return False, f"commit failed: {pdata.get('message', pstatus)}"


def commit_text(text: str, repo_spec: str, branch: str, file_path: str,
                token: str, message: str) -> tuple[bool, str]:
    return commit_bytes(text.encode("utf-8"), repo_spec, branch, file_path, token, message)
