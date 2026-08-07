# Dev server for the slides: rebuild on save + live-reload in the browser.
#
#     julia slides/serve.jl [port]      # default 8383
#
# Open http://localhost:8383/ and edit slides.md / template.html / data/*.js —
# the browser reloads on save, staying on the current slide. A build error
# shows up as an overlay on the slide (with the slide number) instead of
# killing the server; fix the file and it clears itself.
#
# Stdlib only. Opening deck.html straight from file:// also still works,
# just without the auto-reload.

using Sockets

ENV["SLIDES_NO_AUTOBUILD"] = "1"
include(joinpath(@__DIR__, "build.jl"))   # provides build() and DIR

const PORT = isempty(ARGS) ? 8383 : parse(Int, ARGS[1])

const BUILD_STAMP = Ref(0.0)
const BUILD_ERR = Ref("")
const BUILD_LOCK = ReentrantLock()
const BUILDJL = joinpath(@__DIR__, "build.jl")
const BUILDJL_LOADED = Ref(mtime(BUILDJL))

function srcstamp()
    files = [joinpath(DIR, "slides.md"), joinpath(DIR, "template.html"), BUILDJL]
    ddir = joinpath(DIR, "data")
    isdir(ddir) && append!(files, filter(f -> endswith(f, ".js"), readdir(ddir; join = true)))
    return maximum(mtime, files)
end

function maybe_rebuild()
    return lock(BUILD_LOCK) do
        s = srcstamp()
        s == BUILD_STAMP[] && return
        try
            if mtime(BUILDJL) > BUILDJL_LOADED[]   # hot-reload the compiler itself
                Base.include(Main, BUILDJL)
                BUILDJL_LOADED[] = mtime(BUILDJL)
                println("reloaded build.jl")
            end
            Base.invokelatest(Main.build)
            BUILD_ERR[] = ""
        catch e
            BUILD_ERR[] = sprint(showerror, e)
            println(stderr, "build error: ", BUILD_ERR[])
        end
        BUILD_STAMP[] = s
    end
end

const MIME_TYPES = Dict(
    ".html" => "text/html; charset=utf-8",
    ".js" => "text/javascript; charset=utf-8",
    ".css" => "text/css; charset=utf-8",
    ".md" => "text/plain; charset=utf-8",
    ".svg" => "image/svg+xml",
    ".png" => "image/png",
    ".woff2" => "font/woff2",
    ".woff" => "font/woff",
    ".ttf" => "font/ttf",
)

function respond(sock, code, ct, body)
    msg = code == 200 ? "OK" : code == 404 ? "Not Found" : "Error"
    write(
        sock, "HTTP/1.1 $code $msg\r\nContent-Type: $ct\r\n" *
            "Content-Length: $(sizeof(body))\r\nCache-Control: no-store\r\n" *
            "Connection: close\r\n\r\n"
    )
    return write(sock, body)
end

function handle(sock)
    return try
        req = readline(sock)
        parts = split(req, ' ')
        length(parts) >= 2 || return
        path = String(first(split(parts[2], '?')))
        while true                       # drain request headers
            l = readline(sock)
            (isempty(l) || l == "\r") && break
        end
        if path == "/__version"
            maybe_rebuild()
            body = isempty(BUILD_ERR[]) ? string(BUILD_STAMP[]) : "error:" * BUILD_ERR[]
            return respond(sock, 200, "text/plain; charset=utf-8", body)
        end
        path == "/" && (path = "/deck.html")
        path == "/deck.html" && maybe_rebuild()
        occursin("..", path) && return respond(sock, 403, "text/plain", "forbidden")
        file = joinpath(DIR, lstrip(path, '/'))
        isfile(file) || return respond(sock, 404, "text/plain", "not found: $path")
        ct = get(MIME_TYPES, lowercase(splitext(file)[2]), "application/octet-stream")
        respond(sock, 200, ct, read(file))
    catch
    finally
        close(sock)
    end
end

server = listen(Sockets.localhost, PORT)
println("slides: http://localhost:$PORT/  (Ctrl-C to stop)")
maybe_rebuild()
while true
    sock = accept(server)
    errormonitor(@async handle(sock))
end
