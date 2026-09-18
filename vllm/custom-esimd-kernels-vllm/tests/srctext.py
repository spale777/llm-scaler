"""Source-text helpers for the static contract tests.

A substring assertion over raw source is defeated three ways: the token moves
into a comment, the spacing changes, or a string literal contains it. Every
assertion over a .h/.sycl/.cc file goes through `code()`, which closes all three.
"""

import re


def _strip_comments(src: str) -> str:
    """Remove comments and blank the contents of string and char literals.

    A literal's body is replaced, not preserved, so no assertion is satisfiable
    by parking its phrase in a string; the quotes stay, keeping token adjacency.
    The cases below are quotes that open no literal -- misreading one swallows
    the file and everything asserted over it passes vacuously.
    """
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        # A raw string R"delim( ... )delim" has its own terminator.
        if c == "R" and i + 1 < n and src[i + 1] == '"':
            j = src.find("(", i + 2)
            if j > 0:
                delim = src[i + 2 : j]
                close = ")" + delim + '"'
                k = src.find(close, j + 1)
                out.append('R""')
                i = n if k < 0 else k + len(close)
                continue
        # A digit separator (1'000, 0xFF'FF) -- hex digits included.
        if c == "'" and i > 0 and (src[i - 1].isdigit()
                                   or src[i - 1] in "abcdefABCDEF") \
                and i + 1 < n and src[i + 1].isalnum():
            out.append(c)
            i += 1
            continue
        if c in ('"', "'"):
            quote = c
            out.append(c)
            i += 1
            while i < n:
                if src[i] == "\\":
                    i += 2
                    continue
                if src[i] == quote:
                    out.append(quote)
                    i += 1
                    break
                i += 1
            continue
        # #error / #warning text, where a lone apostrophe ("don't") is prose.
        if c == "#" and (i == 0 or src[i - 1] == "\n"):
            rest = src[i:i + 10]
            if rest.startswith("#error") or rest.startswith("#warning"):
                j = src.find("\n", i)
                i = n if j < 0 else j
                continue
        if c == "/" and i + 1 < n:
            if src[i + 1] == "/":
                j = src.find("\n", i)
                i = n if j < 0 else j
                continue
            if src[i + 1] == "*":
                j = src.find("*/", i + 2)
                i = n if j < 0 else j + 2
                out.append(" ")
                continue
        out.append(c)
        i += 1
    return "".join(out)


def code(src: str) -> str:
    """Comment-free, whitespace-normalised source for substring assertions."""
    return re.sub(r"\s+", " ", _strip_comments(src)).strip()


def tokens(src: str) -> str:
    """Comment-free source with all whitespace removed.

    Use when the assertion must not depend on spacing at all, e.g. matching
    `block_load<int32_t,32>` regardless of how the author spaced the commas.
    """
    return re.sub(r"\s+", "", _strip_comments(src))


# Statically-false gate forms. `if constexpr (false)` and `while (false)` are
# included because they disable a block just as effectively as `if (false)`.
_DEAD_GATE = re.compile(
    r"(?:if\s*constexpr\s*|if\s*|while\s*)\(\s*"
    r"(?:false|0|0u|0U|0UL|0ULL)\s*(?:&&[^)]*)?\)")


def assert_live(c: str, needle: str, msg: str = "") -> None:
    """The construct must be present AND not in a statically-dead block.

    `if (false) { NEEDLE }` preserves the needle verbatim, so the scan
    brace-walks outward over every open guard rather than matching a fixed
    lookback, which sees only the braceless form. `c` must be `code()`-normalised.
    """
    assert needle in c, msg or f"missing: {needle}"
    for m in re.finditer(re.escape(needle), c):
        head = c[:m.start()]
        # Collect the conditions of every block still open at the needle.
        depth, chain = 0, []
        for t in reversed(list(re.finditer(
                r"\}|(?:if\s*constexpr\s*|if\s*|while\s*)\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*\{|\{",
                head))):
            tok = t.group(0)
            if tok == "}":
                depth += 1
            elif depth:
                depth -= 1
            elif t.group(1) is not None:
                chain.append(t.group(1))
        for cond in chain:
            # A dead literal ANYWHERE in a top-level && chain kills the block.
            parts, depth, cur = [], 0, ""
            k = 0
            while k < len(cond):
                ch = cond[k]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                if depth == 0 and cond[k:k + 2] == "&&":
                    parts.append(cur)
                    cur = ""
                    k += 2
                    continue
                cur += ch
                k += 1
            parts.append(cur)
            def _statically_false(q):
                q = q.strip()
                if re.fullmatch(r"\(*\s*(?:false|0|0u|0U|0UL|0ULL)\s*\)*", q):
                    return True
                # Fold a comparison of two integer literals: `(1 == 2)` reads
                # as ordinary code and disables a block as well as `false`.
                # int(x, 0) so 0x1, 0b1 and 0755 fold too. A NAMED constexpr
                # false is not detectable here -- a real limit.
                m = re.fullmatch(
                    r"\(*\s*(-?(?:0[xXbB])?[0-9a-fA-F]+)[uUlL]*\s*"
                    r"(==|!=|<|<=|>|>=)\s*"
                    r"(-?(?:0[xXbB])?[0-9a-fA-F]+)[uUlL]*\s*\)*", q)
                if not m:
                    return False
                try:
                    a, b = int(m.group(1), 0), int(m.group(3), 0)
                except ValueError:
                    return False
                op = m.group(2)
                return not {"==": a == b, "!=": a != b, "<": a < b,
                            "<=": a <= b, ">": a > b, ">=": a >= b}[op]

            assert not any(_statically_false(q) for q in parts), (
                f"{msg or needle}: present but inside a statically-dead block "
                f"guarded by ({cond})"
            )
        # A braceless dead gate immediately before the needle.
        assert not _DEAD_GATE.search(head[-64:] if len(head) > 64 else head) \
            or "{" in head[-64:], (
            f"{msg or needle}: present but disabled by a dead gate -> "
            f"...{head[-48:]}"
        )


def schema_of(raw: str, op: str) -> str:
    """Return the torch schema string for `op` from RAW source.

    Deliberately not `code()`, which blanks literal bodies and would erase the
    schema. Raises if the op is absent: a schema test that silently skips a
    missing op checks nothing.
    """
    # C++ concatenates "esimd_foo" "(Tensor ..." into one schema, so join
    # adjacent literals before looking for the op name.
    joined = re.sub(r'"\s*"', "", raw)
    i = joined.find('"' + op + '(')
    if i < 0:
        raise AssertionError(f"schema for {op} not found")
    # Honour backslash escapes so a schema containing \" is not truncated.
    j, n = i + 1, len(joined)
    while j < n:
        if joined[j] == "\\":
            j += 2
            continue
        if joined[j] == '"':
            break
        j += 1
    if j >= n:
        raise AssertionError(f"unterminated schema for {op}")
    return joined[i:j + 1]


def patch_code(raw: str) -> str:
    """Patch text with Python `#` comment tails removed.

    The `code()` equivalent for integration patches: a comment quoting a guard
    must never satisfy an assertion about the guard. Diff markers (---, +++,
    @@) are preserved, since a `#` inside them is not a comment.
    """
    out = []
    for line in raw.splitlines():
        if line.startswith(("---", "+++", "@@")):
            out.append(line)
            continue
        # Cut at the first UNQUOTED `#`; a false strip only makes a caller's
        # assertion stricter.
        depth_s = depth_d = False
        cut = len(line)
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "\\":
                i += 2
                continue
            if ch == "'" and not depth_d:
                depth_s = not depth_s
            elif ch == '"' and not depth_s:
                depth_d = not depth_d
            elif ch == "#" and not depth_s and not depth_d:
                cut = i
                break
            i += 1
        out.append(line[:cut])
    return "\n".join(out)


def is_vacuous_predicate(expr: str) -> bool:
    """True when a Python gate expression is true with every leaf false.

    Substitute False for every leaf, keep the and/or/not/any/all/bool structure
    and evaluate: still True means no leaf can make the gate refuse anything. A
    syntactic ban on `or`/`True` is both too strict, since there are legitimate
    disjunctions, and too weak, since `any((<predicate>, 1))` carries neither.
    """
    import ast as _ast

    try:
        tree = _ast.parse(expr.strip(), mode="eval").body
    except SyntaxError as exc:
        # NOT `return False`: an over-capturing caller would read a syntax
        # error as "not vacuous" and every mutation would pass.
        raise AssertionError(
            f"is_vacuous_predicate could not parse {expr[:120]!r}: {exc}. "
            "The caller's extraction is wrong; a parse failure must not read "
            "as a clean predicate."
        ) from None

    def ev(n):
        if isinstance(n, _ast.BoolOp):
            vals = [ev(v) for v in n.values]
            return any(vals) if isinstance(n.op, _ast.Or) else all(vals)
        if isinstance(n, _ast.UnaryOp) and isinstance(n.op, _ast.Not):
            return not ev(n.operand)
        if isinstance(n, _ast.Constant):
            return bool(n.value)
        # A non-empty container literal is TRUTHY at runtime; the "opaque ->
        # False" default would read `any((expr, [1]))` as not vacuous.
        if isinstance(n, (_ast.List, _ast.Tuple, _ast.Set)):
            return bool(n.elts)
        if isinstance(n, _ast.Dict):
            return bool(n.keys)
        if isinstance(n, _ast.Call):
            fn = getattr(n.func, "id", "")
            if fn in ("any", "all"):
                args = n.args[0] if n.args else None
                if isinstance(args, (_ast.Tuple, _ast.List, _ast.Set)):
                    vals = [ev(e) for e in args.elts]
                    return (any(vals) if fn == "any" else all(vals))
                return False
            if fn == "bool" and n.args:
                return ev(n.args[0])
            return False          # an opaque call is a leaf
        if isinstance(n, (_ast.Compare, _ast.Name, _ast.Attribute,
                          _ast.Subscript)):
            return False          # leaves
        return False

    return ev(tree)


def assert_single_write(src: str, name: str, msg: str = "") -> None:
    """`name` must be assigned exactly once in `src`.

    Pinning an expression's TEXT leaves its OPERANDS free: writing to one
    before the pinned line and restoring it after keeps the text byte-identical
    and the guard inert. An exact-RHS pin is only as strong as the write counts
    of the identifiers it reads.
    """
    import re as _re
    esc = _re.escape(name)
    # Plain writes, writes THROUGH A CAST (`const_cast<int&>(t1) = ...`), and
    # compound writes, which substitute just as well. A loop's own
    # `for (...; ...; x += n)` header is excluded below by requiring the
    # compound write to be statement-leading.
    n = len(_re.findall(rf"(?<![+\-*/%&|^!<>=]){esc}\s*=(?![=])", src))
    n += len(_re.findall(rf"_cast<[^>]*>\(\s*{esc}\s*\)\s*=(?![=])", src))
    body = _re.sub(r"for\s*\([^;]*;[^;]*;[^)]*\)", " ", src)
    # `)` joins the [;{}] anchor to catch the body of a braceless if/while.
    n += len(_re.findall(
        rf"(?:^|[;{{}}\)])\s*{esc}\s*(?:\+|-|\*|/|%|&|\||\^|<<|>>)=",
        body, _re.M))
    # ++/-- in either position.
    n += len(_re.findall(rf"(?:\+\+|--)\s*{esc}\b", body))
    n += len(_re.findall(rf"\b{esc}\s*(?:\+\+|--)", body))
    assert n == 1, (
        f"{msg or name}: assigned {n} times; an exact-text pin on an "
        f"expression that reads {name} is inert if {name} itself can be "
        "rewritten around it"
    )
