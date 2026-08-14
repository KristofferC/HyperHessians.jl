# Slides

This slide setup also exists as a standalone template repo —
[github.com/KristofferC/deckmd](https://github.com/KristofferC/deckmd)
([live demo](https://kristofferc.github.io/deckmd/)) — start new decks from
there rather than copying files out of here.

## The loop

```
julia slides/serve.jl        # then open http://localhost:8383/
```

Edit **`slides.md`** and save — the browser rebuilds and reloads itself,
staying on the slide you're looking at. A syntax error shows up as an overlay
on the slide (with the slide number); fix it and the overlay clears. The
server also picks up edits to `template.html` and `data/*.js`.

Without the server: `julia slides/build.jl` (or `jld run slides/build.jl`)
and reload `deck.html` by hand — it works straight from `file://`, fully
offline, which is also how you present it.

| file | role |
| --- | --- |
| `slides.md` | the content you edit — slides separated by `---` |
| `serve.jl` | dev server: rebuild-on-save + live reload (stdlib only) |
| `build.jl` | Markdown → HTML compiler (no dependencies) |
| `template.html` | all CSS + the deck JS engine — edit to change the look |
| `data/*.js` | figure data — regenerate with `julia slides/data/generate.jl` |
| `vendor/katex/` | vendored KaTeX so `$math$` works offline |
| `deck.html` | **generated** output — never edit |

## Syntax

Slides are separated by `---`. Within a slide (colons after `@keys` optional):

```markdown
@eyebrow Component · table          small caps line above the heading
# Measured totals                   the heading
@kicker A muted subtitle            under the heading
@layout title | center              special layouts (title page, centered)
@chips a | b | c                    pill chips row
@keys <kbd>→</kbd> next             small footer line

+ stepped bullet (fragment)         - always-visible bullet
  - nested bullet (indent a - or +)
  ~ muted sub-line for the bullet above

**bold** *italic* `code` ==highlight==
$\varepsilon_1^2 = 0$ inline and $$ H_{ij} = \dots $$ display math (KaTeX)

| a | b |                           pipe table, auto-wrapped in a panel
| --- | --- |
| x | ==2.3×== |                    ==cell== colors the payoff column
?> caption text                     figure caption (after tables/figures)

!big 4.39×                          giant number

@gap                                vertical spacer (24px); @gap 40 or @gap 2em
                                    for a specific size

```julia title="..." sub="..."      code card; also ```diff and ```julia>
hessian!(H, f, x, cfg)  #!hl          #!hl spotlights a line
```                                 (julia> renders REPL prompts)
@pills good:0 B | bad:slow | note   verdict pills right after a fence
```

`lang | some caption` inside a fence info line is shorthand for `title="..."`.

Layout containers:

```markdown
::: cols            two columns, split by  :: col  (or +++)
::: panel Title     boxed panel with an optional title
::: fragment        click-to-reveal block (steps like a + bullet)
:::                 closes a container
```

Raw HTML is the escape hatch: any line starting with `<` passes through, and
`~~~ … ~~~` fences pass whole blocks through untouched (see the tiles and
diagram slides). Everything above is implemented in `build.jl` — it's ~400
lines of plain Julia, so extending the syntax is fair game.

## Presenting

`→`/`←` navigate (fragments step first), `f` fullscreen, `t` light/dark,
`Home`/`End` jump, the URL hash deep-links a slide, and printing gives one
slide per page (PDF export). On touch screens: tap the right/left edge or
swipe to navigate. `deck.html` + `data/` + `vendor/` is all you
need on the presentation machine — no network.

`julia slides/build.jl --pdf` emits **`deck.pdf`** (one slide per page, light
theme, fragments shown) via headless Chrome — the scripted version of
printing the deck from the browser.

`julia slides/build.jl --single` additionally emits **`deck-single.html`**:
the same deck with KaTeX, its fonts (woff2, as data: URIs), and the figure
data all inlined — one ~700 KB file with no other requirements, for mailing
or hosting anywhere.
