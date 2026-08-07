# Build slides/deck.html from slides/slides.md + slides/template.html.
#
#     julia slides/build.jl          # or: jld run slides/build.jl
#
# slides.md syntax
# ----------------
#   ---                    slide separator (line by itself)
#   # Heading              slide heading (h1 on title layout, else h2)
#   @layout title|center   optional layout (default: heading top, body below)
#   @eyebrow text          small caps line above the heading   (colon after
#   @kicker text           muted line under the heading         the key is
#   @chips a | b | c       pill chips row                       optional)
#   @keys html             small footer line (raw HTML allowed)
#   - bullet               always-visible bullet        (ul.points)
#   + bullet               stepped bullet (.fragment)
#     ~ sub line           muted sub-line for the bullet above
#   **b** *i* `c` ==mark== inline formatting · [text](url) links
#   :github: :mail:        inline icons (octicons, follow text color)
#   $x^2$ and $$ x^2 $$    LaTeX math (KaTeX, vendored in slides/vendor)
#   | a | b |              pipe table -> styled table in a panel
#   ==2.3×==               table cell rendered as the accent .win column
#   ?> text                figure caption (also directly after a table)
#   !big 4.39×             giant number
#   ```julia title="..." sub="..." size="16"   code card (julia | julia> | diff |
#                          anything); size overrides the 19px code font (px or any CSS unit)
#   ```julia> | caption              REPL block; `lang | text` is title shorthand
#   ```diff2                         side-by-side diff: - lines left, + lines right
#   ... #!hl               highlighted line inside a code fence
#   @pills good:a | bad:b | c        verdict pills, directly after a fence
#   ::: panel [title] ... :::        panel box (optional title)
#   ::: cols ... :::       columns, split by `:: col` (or `+++`) lines
#   ~~~ ... ~~~            raw HTML passthrough (escape hatch)
#   <div ...>              a line starting with < is also passed through raw
#   @fig lane-hh           generated figure (lane-hh | lane-fd | lane-hh4 |
#                          lane-jet4 | lane-jet | lane-legend): memory-layout cell strips
#   @fig cells v = value | e1*8 = gradient   ad-hoc strip: groups split by |,
#                          cells are class[*count] tokens, optional = label;
#                          also works mid-text (e.g. in a bullet) — then cells
#                          shrink to text height and labels sit beside them;
#                          bare @fig runs to end of line — use @fig{...} to
#                          delimit it, e.g. two figs on one line

const DIR = @__DIR__

# ---------------- utilities ----------------

escape_html(s) = replace(s, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")

# Replace regex matches with \x01N\x02 placeholders so later passes skip them.
function stash(s::AbstractString, re::Regex, toks::Vector{String}, wrap)
    io = IOBuffer()
    pos = firstindex(s)
    for m in eachmatch(re, s)
        m.offset < pos && continue
        print(io, SubString(s, pos, prevind(s, m.offset)))
        push!(toks, wrap(m))
        print(io, '\x01', length(toks), '\x02')
        pos = m.offset + ncodeunits(m.match)
    end
    pos <= lastindex(s) && print(io, SubString(s, pos, lastindex(s)))
    return String(take!(io))
end

function restore(s::AbstractString, toks::Vector{String})
    while occursin('\x01', s)
        s = replace(s, r"\x01(\d+)\x02" => m -> toks[parse(Int, m[2:(lastindex(m) - 1)])])
    end
    return s
end

# ---------------- inline markdown ----------------

# Inline icons (octicons), filled with the surrounding text color
const ICON_PATHS = Dict(
    "github" => "M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z",
    "mail" => "M1.75 2h12.5c.966 0 1.75.784 1.75 1.75v8.5A1.75 1.75 0 0 1 14.25 14H1.75A1.75 1.75 0 0 1 0 12.25v-8.5C0 3.784.784 2 1.75 2ZM1.5 12.251c0 .138.112.25.25.25h12.5a.25.25 0 0 0 .25-.25V5.809L8.38 9.397a.75.75 0 0 1-.76 0L1.5 5.809v6.442Zm13-8.181v-.32a.25.25 0 0 0-.25-.25H1.75a.25.25 0 0 0-.25.25v.32L8 7.88Z",
)
icon_svg(name) = "<svg viewBox=\"0 0 16 16\" aria-label=\"$name\"><path d=\"$(ICON_PATHS[name])\"/></svg>"

function inline_md(s::AbstractString)
    toks = String[]
    # inline figures: `@fig{...}` is delimited (several per line); a bare
    # `@fig ...` consumes the rest of the line (see render_figure_inline)
    s = stash(s, r"@fig\{([^{}]*)\}", toks, m -> render_figure_inline(strip(m.captures[1])))
    s = stash(s, r"@fig:?\s+(.+)$", toks, m -> render_figure_inline(strip(m.captures[1])))
    s = stash(s, r":(github|mail):", toks, m -> icon_svg(m.captures[1]))
    s = stash(s, r"`([^`]+)`", toks, m -> "<code>" * escape_html(m.captures[1]) * "</code>")
    # LaTeX math passes through untouched (KaTeX renders it in the browser)
    s = stash(s, r"\$\$.+?\$\$", toks, m -> escape_html(m.match))
    s = stash(s, r"\$[^\$\s](?:[^\$]*[^\$\s])?\$", toks, m -> escape_html(m.match))
    s = replace(s, r"\[([^\]]+)\]\(([^)]+)\)" => SubstitutionString("<a href=\"\\2\">\\1</a>"))
    s = replace(s, r"\*\*(.+?)\*\*" => s"<b>\1</b>")
    s = replace(s, r"==(.+?)==" => s"<mark>\1</mark>")
    s = replace(s, r"(?<![\w*])\*([^\s*][^*]*)\*(?!\*)" => s"<i>\1</i>")
    return restore(s, toks)
end

# ---------------- julia syntax highlighting ----------------

const JL_KW = [
    "using", "import", "export", "module", "function", "return",
    "struct", "mutable", "const", "macro", "quote", "local", "global",
    "for", "in", "while", "if", "elseif", "else", "begin", "let",
    "do", "try", "catch", "finally", "end",
]
const JL_KW_RE = Regex("(?<![\\w!.@\"])(?:" * join(JL_KW, "|") * ")(?![\\w!])")

function hl_julia(line::AbstractString)
    toks = String[]
    s = escape_html(line)
    s = stash(s, r"\"(?:[^\"\\]|\\.)*\"", toks, m -> "<span class=\"s\">" * m.match * "</span>")
    s = stash(s, r"#.*$", toks, m -> "<span class=\"cm\">" * m.match * "</span>")
    s = stash(s, r"@[A-Za-z_][\w!]*", toks, m -> "<span class=\"fn\">" * m.match * "</span>")
    s = stash(s, JL_KW_RE, toks, m -> "<span class=\"k\">" * m.match * "</span>")
    s = stash(s, r"(?<![\w!.\x01])[A-Za-z_][\w!]*(?=\()", toks, m -> "<span class=\"fn\">" * m.match * "</span>")
    s = stash(s, r"(?<![\w.\x01])\d[\d_]*(?:\.\d+)?(?:[ef][+-]?\d+)?", toks, m -> "<span class=\"n\">" * m.match * "</span>")
    return restore(s, toks)
end

# ---------------- block renderers ----------------

function render_pills(spec)
    items = String[]
    for p in split(spec, '|')
        p = strip(p)
        isempty(p) && continue
        cls = "v-pill"
        for tag in ("good", "bad")
            if startswith(p, tag * ":")
                cls *= " " * tag
                p = strip(p[(length(tag) + 2):end])
                break
            end
        end
        push!(items, "<span class=\"" * cls * "\">" * escape_html(p) * "</span>")
    end
    return "<div class=\"cc-verdict\">\n" * join(items, "\n") * "\n</div>"
end

# token-level line diff (LCS over words/whitespace): returns per side a vector
# of (text, changed) chunks covering the line; only differing tokens are marked
function line_chunks(a, b)
    tok(s) = [String(m.match) for m in eachmatch(r"\s+|\S+", s)]
    ta, tb = tok(a), tok(b)
    na, nb = length(ta), length(tb)
    L = zeros(Int, na + 1, nb + 1)
    for i in na:-1:1, j in nb:-1:1
        L[i, j] = ta[i] == tb[j] ? L[i + 1, j + 1] + 1 : max(L[i + 1, j], L[i, j + 1])
    end
    ca, cb = Tuple{String, Bool}[], Tuple{String, Bool}[]
    add!(v, s, ch) = (!isempty(v) && v[end][2] == ch) ?
        (v[end] = (v[end][1] * s, ch)) : push!(v, (s, ch))
    i = j = 1
    while i <= na || j <= nb
        if i <= na && j <= nb && ta[i] == tb[j]
            add!(ca, ta[i], false)
            add!(cb, tb[j], false)
            i += 1; j += 1
        elseif j > nb || (i <= na && L[i + 1, j] >= L[i, j + 1])
            add!(ca, ta[i], true)
            i += 1
        else
            add!(cb, tb[j], true)
            j += 1
        end
    end
    return ca, cb
end

# two-column diff: `-` lines left, `+` lines right, context lines on both sides;
# within a hunk the shorter side is padded with blank lines
function render_split_diff(body, presty)
    lft = Union{String, Nothing}[]
    rgt = Union{String, Nothing}[]
    minus = String[]
    plus = String[]
    function flushrun!()
        for k in 1:max(length(minus), length(plus))
            push!(lft, k <= length(minus) ? minus[k] : nothing)
            push!(rgt, k <= length(plus) ? plus[k] : nothing)
        end
        empty!(minus)
        return empty!(plus)
    end
    for raw in split(body, '\n')
        c = isempty(raw) ? ' ' : raw[1]
        if c == '-'
            isempty(plus) || flushrun!()   # a `-` after `+`s starts a new hunk
            push!(minus, String(raw))
        elseif c == '+'
            push!(plus, String(raw))
        else
            flushrun!()
            push!(lft, String(raw))
            push!(rgt, String(raw))
        end
    end
    flushrun!()
    # a row whose - and + sides are identical didn't actually change: show it uncolored
    for k in eachindex(lft)
        l, r = lft[k], rgt[k]
        if l !== nothing && r !== nothing && startswith(l, "-") && startswith(r, "+") &&
                l[2:end] == r[2:end]
            lft[k] = rgt[k] = l[2:end]
        end
    end
    # paired changed rows get their differing tokens emphasized (.seg)
    hl(chunks) = join(
        map(chunks) do (s, ch)
            ch ? "<span class=\"seg\">" * escape_html(s) * "</span>" : escape_html(s)
        end
    )
    lspans, rspans = String[], String[]
    for k in eachindex(lft)
        l, r = lft[k], rgt[k]
        if l !== nothing && r !== nothing && startswith(l, "-") && startswith(r, "+")
            lp, rp = line_chunks(l[2:end], r[2:end])
            for (p, cls, spans) in ((lp, "bad", lspans), (rp, "good", rspans))
                s = hl(p)
                isempty(s) && (s = "&nbsp;")
                push!(spans, "<span class=\"ln " * cls * "\">" * s * "</span>")
            end
        else
            for (v, spans) in ((l, lspans), (r, rspans))
                cls = v !== nothing && startswith(v, "-") ? "ln bad" :
                    v !== nothing && startswith(v, "+") ? "ln good" : "ln"
                cls != "ln" && (v = v[2:end])   # color carries the +/- meaning
                content = (v === nothing || isempty(v)) ? "&nbsp;" : escape_html(v)
                push!(spans, "<span class=\"" * cls * "\">" * content * "</span>")
            end
        end
    end
    return "<div class=\"cc-split\">\n<pre" * presty * "><code>" * join(lspans, "") *
        "</code></pre>\n<pre" * presty * "><code>" * join(rspans, "") *
        "</code></pre>\n</div>\n"
end

function render_codecard(lang, attrs, body, pills)
    out = String[]
    for raw in split(body, '\n')
        hl = occursin("#!hl", raw)
        raw = replace(raw, r"\s*#!hl" => "")
        cls = "ln" * (hl ? " hl" : "")
        if lang == "diff"
            c = isempty(raw) ? ' ' : raw[1]
            cls = c == '-' ? "ln bad" : c == '+' ? "ln good" : cls
            content = escape_html(raw)
        elseif lang == "julia"
            content = hl_julia(raw)
        elseif lang == "julia>" || lang == "repl"
            m = match(r"^(\s*)julia>(.*)$", raw)
            if m !== nothing
                content = m.captures[1] * "<span class=\"prompt\">julia&gt;</span>" * hl_julia(m.captures[2])
            else
                content = hl_julia(raw)
            end
        else
            content = escape_html(raw)
        end
        # an empty block-level span has zero height — keep blank lines visible
        isempty(content) && (content = "&nbsp;")
        push!(out, "<span class=\"" * cls * "\">" * content * "</span>")
    end
    title = get(attrs, "title", nothing)
    sub = get(attrs, "sub", nothing)
    size = get(attrs, "size", nothing)   # font-size override; bare number = px
    presty = size === nothing ? "" :
        " style=\"font-size: " * escape_html(occursin(r"^[\d.]+$", size) ? size * "px" : size) * "\""
    io = IOBuffer()
    print(io, "<div class=\"codecard\">\n")
    if title !== nothing || sub !== nothing
        print(io, "  <div class=\"cc-head\">\n")
        title !== nothing && print(io, "    <span class=\"cc-title\">", inline_md(title), "</span>\n")
        sub !== nothing && print(io, "    <span class=\"cc-sub\">", escape_html(sub), "</span>\n")
        print(io, "  </div>\n")
    end
    # .ln spans are display:block — no newlines between them, or <pre> renders
    # each one as an extra blank line box (double spacing)
    if lang == "diff2"
        print(io, render_split_diff(body, presty))
    else
        print(io, "<pre", presty, "><code>", join(out, ""), "</code></pre>\n")
    end
    pills !== nothing && print(io, render_pills(pills), "\n")
    print(io, "</div>")
    return String(take!(io))
end

# `cells` spec: groups split by |, cells are class[*count] tokens, optional = label
function parse_cellgroups(spec)
    groups = Tuple{Vector{String}, String}[]
    for g in split(spec, '|')
        sp, label = occursin('=', g) ? strip.(split(g, '=', limit = 2)) : (strip(g), "")
        cells = String[]
        for tok in split(sp)
            m = match(r"^([a-z]\w*)(?:\*(\d+))?$", tok)
            m === nothing && error("bad cell token in @fig cells: $tok")
            append!(
                cells, fill(
                    String(m.captures[1]),
                    m.captures[2] === nothing ? 1 : parse(Int, m.captures[2])
                )
            )
        end
        push!(groups, (cells, String(label)))
    end
    return groups
end

# @fig inside running text (bullets, paragraphs): cells at text height, labels beside
function render_figure_inline(spec)
    if (cm = match(r"^cells\s+(.*)$", spec)) !== nothing
        parts = map(parse_cellgroups(cm.captures[1])) do (cells, label)
            g = "<span class=\"cellgroup\">" *
                join("<i class=\"cell " * c * "\"></i>" for c in cells) * "</span>"
            isempty(label) ? "<span>" * g * "</span>" :
                "<span>" * g * " " * inline_md(label) * "</span>"
        end
        return "<span class=\"fig-inline\">" * join(parts) * "</span>"
    end
    return "<span class=\"fig-inline\">" * render_figure(spec) * "</span>"
end

# memory-layout strips: one colored cell per Float64 slot, grouped as stored
function render_figure(name)
    cell(c) = "<div class=\"cell " * c * "\"></div>"
    group(cs) = "<div class=\"cellgroup\">" * join(cell.(cs)) * "</div>"
    if name == "lane-legend"
        items = [
            ("value", "v", "v"),
            ("gradient chunk 1 (8)", "e1", "f1"),
            ("gradient chunk 2 (8)", "e2", "f2"),
            ("Hessian block (8×8)", "e12", "f3"),
        ]
        spans = map(items) do (label, chh, cfd)
            "<span><i class=\"cell " * chh * "\"></i><i class=\"cell " * cfd *
                "\"></i> " * escape_html(label) * "</span>"
        end
        return "<div class=\"legend\">\n" * join(spans, "\n") * "\n</div>"
    end
    if (cm = match(r"^cells\s+(.*)$", name)) !== nothing
        parts = String[]
        labeled = false
        for (cells, label) in parse_cellgroups(cm.captures[1])
            if isempty(label)
                push!(parts, group(cells))
            else
                labeled = true
                push!(
                    parts, "<div class=\"cellcol\">" * group(cells) *
                        "<div class=\"cell-label\">" * inline_md(label) * "</div></div>"
                )
            end
        end
        labeled && return "<div class=\"lane-strip labeled\">\n" * join(parts, "\n") * "\n</div>"
        return "<div class=\"lane-strip\">\n" * join(parts, "\n") * "\n</div>"
    end
    groups = String[]
    if name == "lane-hh"          # HyperDual{8,8}: each group contiguous
        push!(groups, group(["v"]), group(fill("e1", 8)), group(fill("e2", 8)))
        foreach(_ -> push!(groups, group(fill("e12", 8))), 1:8)
    elseif name == "lane-fd"      # nested Dual 8×8: gradient interleaved into rows
        push!(groups, group(["v"; fill("f1", 8)]))
        foreach(_ -> push!(groups, group(["f2"; fill("f3", 8)])), 1:8)
    elseif name == "lane-hh4"     # HyperDual{4,4}
        push!(groups, group(["v"]), group(fill("e1", 4)), group(fill("e2", 4)))
        foreach(_ -> push!(groups, group(fill("e12", 4))), 1:4)
    elseif name == "lane-jet4"    # Jet{4}: upper triangle only
        push!(groups, group(["v"]), group(fill("e1", 4)))
        foreach(r -> push!(groups, group(fill("e12", 4 - r))), 0:3)
    elseif name == "lane-jet"     # Jet{8}: upper triangle only
        push!(groups, group(["v"]), group(fill("e1", 8)))
        foreach(r -> push!(groups, group(fill("e12", 8 - r))), 0:7)
    else
        error("unknown figure: @fig $name")
    end
    return "<div class=\"lane-strip\">\n" * join(groups, "\n") * "\n</div>"
end

function split_row(line)
    s = strip(line)
    startswith(s, "|") && (s = s[2:end])
    endswith(s, "|") && (s = s[1:prevind(s, lastindex(s))])
    return strip.(split(s, '|'))
end

function render_table(rows, caption)
    length(rows) >= 2 || error("table needs a header and separator row")
    header = split_row(rows[1])
    body = [split_row(r) for r in rows[3:end]]
    io = IOBuffer()
    print(io, "<div class=\"panel\">\n<table>\n<thead><tr>")
    for h in header
        print(io, "<th>", inline_md(h), "</th>")
    end
    print(io, "</tr></thead>\n<tbody>\n")
    for row in body
        print(io, "<tr>")
        for cell in row
            m = match(r"^==(.*)==$", cell)
            if m !== nothing
                print(io, "<td class=\"win\">", inline_md(m.captures[1]), "</td>")
            else
                print(io, "<td>", inline_md(cell), "</td>")
            end
        end
        print(io, "</tr>\n")
    end
    print(io, "</tbody>\n</table>\n")
    caption !== nothing && print(io, "<p class=\"fig-caption\">", inline_md(caption), "</p>\n")
    print(io, "</div>")
    return String(take!(io))
end

function render_list(items)
    lis = map(items) do (frag, text, sub)
        s = frag ? "<li class=\"fragment\">" : "<li>"
        s *= inline_md(text)
        sub !== nothing && (s *= "\n    <span class=\"sub\">" * inline_md(sub) * "</span>")
        s * "</li>"
    end
    return "<ul class=\"points\">\n" * join(lis, "\n") * "\n</ul>"
end

# ---------------- content parser ----------------

const BLOCK_PREFIXES = ("```", "~~~", ":::", "|", "?>", "!big ", "- ", "+ ", "<", "@")

startsblock(st) = any(p -> startswith(st, p), BLOCK_PREFIXES)

function render_content(lines)
    out = String[]
    i, n = 1, length(lines)
    while i <= n
        line = lines[i]
        st = strip(line)
        if isempty(st)
            i += 1
        elseif startswith(st, "```")
            info = strip(st[4:end])
            attrs = Dict(
                m.captures[1] => m.captures[2]
                    for m in eachmatch(r"(\w+)=\"([^\"]*)\"", info)
            )
            head = strip(replace(info, r"(\w+)=\"([^\"]*)\"" => ""))
            barm = match(r"^([^|]*)\|(.*)$", head)   # `lang | title` shorthand
            if barm !== nothing
                haskey(attrs, "title") || (attrs["title"] = strip(barm.captures[2]))
                head = strip(barm.captures[1])
            end
            langm = match(r"^(\S+)", head)
            lang = langm === nothing ? "" : String(langm.captures[1])
            j = i + 1
            buf = String[]
            while j <= n && strip(lines[j]) != "```"
                push!(buf, lines[j]); j += 1
            end
            j <= n || error("unterminated ``` fence: $info")
            i = j + 1
            pills = nothing
            k = i
            while k <= n && isempty(strip(lines[k]))
                k += 1
            end
            if k <= n && (pm = match(r"^@pills:?\s+(.*)$", strip(lines[k]))) !== nothing
                pills = strip(pm.captures[1])
                i = k + 1
            end
            push!(out, render_codecard(lang, attrs, join(buf, "\n"), pills))
        elseif st == "~~~"
            j = i + 1
            buf = String[]
            while j <= n && strip(lines[j]) != "~~~"
                push!(buf, lines[j]); j += 1
            end
            j <= n || error("unterminated ~~~ fence")
            push!(out, join(buf, "\n"))
            i = j + 1
        elseif (cm = match(r"^:::\s*(\w+)\s*(.*)$", st)) !== nothing
            name = cm.captures[1]
            extra = strip(cm.captures[2])
            name in ("cols", "panel") || error("unknown container ::: $name")
            depth = 1
            j = i + 1
            buf = String[]
            while j <= n
                s2 = strip(lines[j])
                if match(r"^:::\s*\w+", s2) !== nothing
                    depth += 1
                elseif s2 == ":::"
                    depth -= 1
                    depth == 0 && break
                end
                push!(buf, lines[j]); j += 1
            end
            depth == 0 || error("unterminated ::: $name container")
            if name == "cols"
                parts = Vector{String}[String[]]
                pdepth = 0
                for l in buf
                    s2 = strip(l)
                    if match(r"^:::\s*\w+", s2) !== nothing
                        pdepth += 1
                    elseif s2 == ":::"
                        pdepth -= 1
                    end
                    if pdepth == 0 && (s2 == "+++" || match(r"^::\s*col$", s2) !== nothing)
                        push!(parts, String[])
                    else
                        push!(parts[end], l)
                    end
                end
                # `:: col` directly after `::: cols` leaves an empty leading part
                length(parts) > 1 && all(isempty ∘ strip, parts[1]) && popfirst!(parts)
                cells = ["<div>\n" * render_content(p) * "\n</div>" for p in parts]
                push!(out, "<div class=\"cols\">\n" * join(cells, "\n") * "\n</div>")
            elseif isempty(extra)
                push!(out, "<div class=\"panel\">\n" * render_content(buf) * "\n</div>")
            else
                push!(
                    out, "<div class=\"panel has-title\">\n<div class=\"panel-title\">" *
                        inline_md(extra) * "</div>\n<div class=\"panel-body\">\n" *
                        render_content(buf) * "\n</div>\n</div>"
                )
            end
            i = j + 1
        elseif startswith(st, "|")
            j = i
            rows = String[]
            while j <= n && startswith(strip(lines[j]), "|")
                push!(rows, strip(lines[j])); j += 1
            end
            caption = nothing
            if j <= n && startswith(strip(lines[j]), "?>")
                caption = strip(strip(lines[j])[3:end])
                j += 1
            end
            push!(out, render_table(rows, caption))
            i = j
        elseif startswith(st, "?>")
            push!(out, "<p class=\"fig-caption\">" * inline_md(strip(st[3:end])) * "</p>")
            i += 1
        elseif startswith(st, "!big ")
            push!(out, "<div class=\"bignum\">" * inline_md(strip(st[6:end])) * "</div>")
            i += 1
        elseif startswith(st, "- ") || startswith(st, "+ ")
            items = Vector{Any}[]
            while i <= n
                s2 = strip(lines[i])
                if startswith(s2, "- ") || startswith(s2, "+ ")
                    push!(items, Any[startswith(s2, "+ "), strip(s2[3:end]), nothing])
                elseif startswith(s2, "~ ") && !isempty(items)
                    items[end][3] = strip(s2[3:end])
                else
                    break
                end
                i += 1
            end
            push!(out, render_list([(a, b, c) for (a, b, c) in items]))
        elseif (fm = match(r"^@fig:?\s+(.+?)\s*$", st)) !== nothing
            push!(out, render_figure(String(fm.captures[1])))
            i += 1
        elseif startswith(st, "<")
            push!(out, line)   # raw HTML line
            i += 1
        else
            buf = String[]
            while i <= n && !isempty(strip(lines[i])) && !startsblock(strip(lines[i]))
                push!(buf, strip(lines[i])); i += 1
            end
            if isempty(buf)   # unrecognized block-prefixed line: show it as text
                push!(buf, st); i += 1
            end
            push!(out, "<p class=\"text\">" * inline_md(join(buf, " ")) * "</p>")
        end
    end
    return join(out, "\n")
end

# ---------------- slide parser ----------------

function render_slide(text)
    layout = ""
    eyebrow = kicker = keys = heading = nothing
    content = String[]
    infence = false
    fencemark = ""
    for line in split(text, '\n')
        st = strip(line)
        if infence
            st == fencemark && (infence = false)
            push!(content, line)
            continue
        elseif startswith(st, "```") || st == "~~~"
            infence = true
            fencemark = startswith(st, "```") ? "```" : "~~~"
            push!(content, line)
            continue
        end
        dm = match(r"^@(layout|eyebrow|kicker|keys|chips):?\s+(.*)$", st)
        if dm !== nothing
            key, val = dm.captures[1], strip(dm.captures[2])
            if key == "layout"
                layout = val
            elseif key == "eyebrow"
                eyebrow = val
            elseif key == "kicker"
                kicker = val
            elseif key == "keys"
                keys = val
            else  # chips
                chips = strip.(split(val, '|'))
                push!(
                    content, "<div class=\"chips\">" *
                        join(("<span class=\"chip\">" * inline_md(c) * "</span>" for c in chips), "\n") *
                        "</div>"
                )
            end
        elseif startswith(st, "# ") && heading === nothing
            heading = strip(st[3:end])
        else
            push!(content, line)
        end
    end
    body = render_content(content)

    eb = eyebrow === nothing ? "" : "  <p class=\"eyebrow\">" * inline_md(eyebrow) * "</p>\n"
    kk = kicker === nothing ? "" : "  <p class=\"kicker\">" * inline_md(kicker) * "</p>\n"
    ks = keys === nothing ? "" : "  <p class=\"keys\">" * keys * "</p>\n"
    if layout == "title"
        h = heading === nothing ? "" : "  <h1>" * heading * "</h1>\n"
        return "<section class=\"slide layout-title\">\n<div class=\"body\">\n" *
            eb * h * kk * body * "\n</div>\n" * ks * "</section>"
    elseif layout == "center"
        h = heading === nothing ? "" : "  <h2>" * heading * "</h2>\n"
        return "<section class=\"slide layout-center\">\n" * eb * h *
            "<div class=\"body\">\n" * body * "\n</div>\n" * ks * "</section>"
    else
        h = heading === nothing ? "" : "  <h2>" * inline_md(heading) * "</h2>\n"
        return "<section class=\"slide\">\n" * eb * h * kk *
            "<div class=\"body\">\n" * body * "\n</div>\n" * ks * "</section>"
    end
end

function split_slides(src)
    slides = String[]
    cur = String[]
    infence = false
    fencemark = ""
    for line in split(src, '\n')
        st = strip(line)
        if infence
            st == fencemark && (infence = false)
            push!(cur, line)
        elseif startswith(st, "```") || st == "~~~"
            infence = true
            fencemark = startswith(st, "```") ? "```" : "~~~"
            push!(cur, line)
        elseif st == "---"
            push!(slides, join(cur, "\n"))
            cur = String[]
        else
            push!(cur, line)
        end
    end
    push!(slides, join(cur, "\n"))
    return [s for s in slides if !isempty(strip(s))]
end

# ---------------- main ----------------

function build()
    md = read(joinpath(DIR, "slides.md"), String)
    template = read(joinpath(DIR, "template.html"), String)
    slides = split_slides(md)
    rendered = map(enumerate(slides)) do (k, s)
        try
            render_slide(s)
        catch e
            error("slide $k: ", sprint(showerror, e))
        end
    end
    html = join(rendered, "\n\n")
    out = replace(template, "<!-- {{SLIDES}} -->" => html)
    banner = "<!-- GENERATED from slides/slides.md by slides/build.jl — do not edit by hand. -->\n"
    out = replace(out, "<!doctype html>\n" => "<!doctype html>\n" * banner; count = 1)
    write(joinpath(DIR, "deck.html"), out)
    return println("deck.html rebuilt: ", length(slides), " slides")
end

# serve.jl includes this file to get build() without triggering a build here
get(ENV, "SLIDES_NO_AUTOBUILD", "") == "1" || build()
