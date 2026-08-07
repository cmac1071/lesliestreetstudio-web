# Leslie Street Studio — Web Project

## What this is
Static HTML/CSS website for Leslie Street Studio LLC. No frameworks, no build tools,
no dependencies beyond Google Fonts (loaded via CDN). Every page is a plain HTML file.

## Repo structure
```
/
  index.html                    — Home
  apps.html                     — Apps index
  about.html                    — About
  contact.html                  — Contact
  CNAME                         — lesliestreetstudio.com (do not edit)
  sitemap.xml, robots.txt       — Google / search crawlers
  app-ads.txt                   — AdMob authorization (CribVision uses AdMob ads). Do not edit
                                   without checking with Chris — wrong entries break ad payouts.
  tools/                        — Reserved, currently empty. Disallowed in robots.txt
                                   (Disallow: /tools/) — leave that rule in place even though
                                   the directory has no content yet.
  CLAUDE.md                     — This file. NOT currently tracked by git (see "Version control"
                                   below) — it still exists on disk, but confirm it's committed
                                   before assuming a fresh clone will have it.
  apps/
    railroad-solitaire.html
    cribvision.html
    barvision.html
    vandoras.html
    deckvision.html
    saypoint.html
  assets/
    css/
      style.css                 — Single global stylesheet, all brand variables here
    images/
      svg/                      — Primary wordmark assets (SVG, always prefer over PNG)
      *.png, *.jpg              — Fallback wordmark assets + app icons + screenshots
    videos/
      *.mp4                     — Self-hosted promo video (portrait 9:16), served directly, no embeds
```

## Hosting
GitHub Pages on cmac1071/lesliestreetstudio-web.
Custom domain: lesliestreetstudio.com (DNS on Namecheap, already configured).
To deploy: commit and push to main. No build step.

---

## Brand — LOCKED. Do not deviate.

### Colors (CSS variables defined in style.css)
| Variable        | Hex       | Role                                      |
|-----------------|-----------|-------------------------------------------|
| --forest        | #1B4332   | Primary brand, backgrounds, wordmark      |
| --mid-green     | #2D6A4F   | Hover and interactive states              |
| --sage          | #95D5B2   | Light accent, secondary text on dark bg   |
| --mint          | #D8F3DC   | Background fills only, never text         |
| --gold          | #C9A84C   | Accent — one moment per composition only  |
| --charcoal      | #3A3835   | Primary text, dark backgrounds            |
| --parchment     | #DDD5C4   | Page background, light text on dark       |
| --stone         | #6B7280   | Secondary/muted text only                 |

**Never use #000000 or #FFFFFF. Never use gold as a fill or background.
Never use Mint or Sage as text colors.**

### Typography
- **Playfair Display** — all display, headlines, subheadings (Google Fonts)
- **DM Sans** — all body, UI, labels (Google Fonts)
- Both loaded in style.css via @import. Do not add other typefaces.

### Wordmark
SVGs live in `assets/images/svg/`. Always use SVG over PNG.
Use the correct variant for each background context:

| Context                        | File                                      |
|-------------------------------|-------------------------------------------|
| Header (Forest Green bg)      | WordmarkStackedParchmentonForest.svg      |
| Footer (Charcoal bg)          | WordmarkHorizontalParchmentonCharcoal.svg |
| Light bg (Parchment)          | WordmarkStackedForestonParchment.svg      |
| Transparent/compositing       | WordmarkStackedParchmentonTransparent.svg |

Do not recreate, reinterpret, or modify the wordmark in any way.

---

## Apps in the App Store

### Railroad Solitaire (v2.1 — live on App Store, released May 25 2026)
- App Store: https://apps.apple.com/app/id6758528011
- Page: apps/railroad-solitaire.html
- Icon: assets/images/RailroadV2_Icon.jpg
- v2.1 Screenshots: RailroadV2_1-TitleScreen.jpg, RailroadV2_1-Cascade.jpg,
  RailroadV2_1-CabooseBonus.jpg, RailroadV2_1-WheelBonus.jpg, RailroadV2_1-Win.jpg,
  RailroadV2_1-Achievements.jpg, RailroadV2_1-Stats.jpg, RailroadV2_1-EndOfGame.jpg
- Legacy v2 screenshots (kept for reference): RailroadV2-Main.jpg, RailroadV2-Win.jpg,
  RailroadV2-CabooseBonus.jpg, RailroadV2-WheelBonus.jpg, RailroadV2-BrandImage.jpg
- Legacy v1 screenshots (kept for reference): Railroadscreenshot.png, Railroadscreenshotwin.png,
  Railroadscreenshotstats.png
- Legacy v1 icon (kept for reference): Railroad_icon.png
- Desktop versions (v1 only): macOS and Windows builds linked from the app page
- GitHub legacy site: https://cmac1071.github.io/RailroadSolitaire/ (redirects to LSS site when live)

### Railroad! Yard Solitaire (Rules Publication)
- Page: apps/railroad-yard-edition.html
- Published 2026-05-31 for copyright establishment purposes (timestamped git commit)
- Standalone official rulebook for the Yard Solitaire variant — playable with a standard 52-card deck
- Contains SVG card illustrations for board setup, suit match, rank match, and cascade examples
- Copyright: Leslie Street Studio LLC © 2026

### CribVision
- App Store: https://apps.apple.com/us/app/cribvision/id6759795942
- Page: apps/cribvision.html
- Icon: assets/images/CribVision_Icon5.png
- Screenshots: CribVisionscreenshotmain.png, CribVisionscreenshotscan.png,
  CribVisionscreenshotscoring.png, CribVisionscreenshotteaching.png
- GitHub legacy site: https://cmac1071.github.io/CribVisionApp/ (redirects to LSS site when live)

### DeckVision
- App Store: https://apps.apple.com/us/app/deckvision/id6762495451
- Page: apps/deckvision.html
- Icon: assets/images/DeckVision_Icon.png
- Screenshots: DeckVisionScreenshot*.jpg

## Apps in Development

### BarVision
- Page: apps/barvision.html
- Icon: assets/images/BarVision_Icon.png

### SayPoint
- Page: apps/saypoint.html
- Support: apps/saypoint/support.html
- Privacy Policy: apps/saypoint/privacy-policy.html
- Icon: assets/images/SayPoint_Icon.png
- Voice-controlled scorekeeper for any physical multiplayer game (cards, dice, darts, etc.). Native iOS 26+/iPadOS 26+, on-device speech recognition (SpeechAnalyzer/SpeechTranscriber, no cloud), push-to-talk or always-listening activation, unlimited concurrent/paused games with permanent history, undo/redo, Low Score Wins/countdown mode. One-time purchase, no ads, no accounts.
- Status: Coming Soon, no release date locked yet — apps.html card and the app page badge both read "Coming Soon." Update to a real App Store link + screenshots once it ships.
- Support and privacy policy pages use the text Chris supplied verbatim (SayPoint_support.md / SayPoint_privacypolicy.md) — do not rewrite the legal copy without his input.

### Vandoras
- Page: apps/vandoras.html
- Support: apps/vandoras/support.html
- Privacy Policy: apps/vandoras/privacy-policy.html
- Icon: assets/images/Vandoras_icon.png
- Release date locked: August 24, 2026 (App Store)
- TestFlight beta open ahead of launch — testers request an invite by emailing support@lesliestreetstudio.com (no public link posted, by design)
- Teaser videos: "Teasers" section on apps/vandoras.html, placed between the feature-keyword ticker and "The Story" section. Self-hosted portrait (9:16) mp4s in assets/videos/, one `.vn-teaser-card` per video inside `.vn-teasers-grid`. Poster frames are extracted stills saved as assets/images/Vandoras_TeaserN_Poster.jpg. Compress per the "Video assets" section above before adding to the repo.
  - Cadence: first published 2026-07-11, new one roughly weekly for 3 more weeks (through early August 2026)
  - Fullscreen/expand uses a single shared `#vn-lightbox` overlay (defined once, right after the Teasers section) — this is the reference implementation of the lightbox pattern described in "Video assets" above. Each teaser card's expand button just calls `openLightbox(source.src, video.poster)`; adding a new teaser doesn't require touching the lightbox itself.
  - Teaser 1: Vandoras_Reel1_TheWorld.mp4, poster Vandoras_Teaser1_Poster.jpg, caption "Vandoras Teaser 1 - The World", dated July 11, 2026
  - To add the next one: duplicate a `.vn-teaser-card` block, swap the video src/poster/caption/date, keep newest-first order

---

## Contact
- App support: support@lesliestreetstudio.com
- General: chris@lesliestreetstudio.com
- Instagram: @lesliestreetstudio
- Facebook: https://facebook.com/lesliestreetstudio

---

## Writing tone
Warm, confident, unhurried. Not precious. Not corporate. Written as a craftsperson.
Short sentences. No superlatives. No startup language.
The tagline is: **Intentional software.**

---

## CSS conventions
- All brand values are CSS variables — never hardcode hex values in HTML or page-level CSS
- Spacing uses --space-* variables, never arbitrary pixel values
- Dark section pattern: add class `on-dark` to the container for correct text colors
- Animations: fade-up class + delay variants (fade-up--delay-1 through delay-4)
- Gold rule separator: `<hr class="gold-rule">` or `<hr class="gold-rule gold-rule--narrow">`
- Section divider ornament: `.section-divider` with inner line/mark/line pattern
- Font sizing: use `var(--size-body)` (16px) for any paragraph/description text a reader
  actually engages with (card descriptions, feature blurbs, section intros). Reserve
  `var(--size-secondary)` (14px) strictly for captions, tags, footnote-level blurbs, and
  metadata. Page-level overrides that default readable content to `--size-secondary` have
  caused real readability complaints (too small at laptop viewing distance) — don't do it
  again.

## Video assets
Self-hosted only — no YouTube/Vimeo embeds, to keep player chrome brand-locked with no
third-party UI. Portrait (9:16) mp4s live in `assets/videos/`.

- **Compression standard, apply to every video before it goes in the repo** (Chris hands off
  full-resolution, high-bitrate masters and keeps his own copies — the web copy should be
  compressed, not the delivered original):
  `ffmpeg -i in.mp4 -vf scale=720:-2 -c:v libx264 -crf 23 -preset slow -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart out.mp4`
  If dark/gradient-heavy footage bands at CRF 23, step down to CRF 20–21 rather than raising
  bitrate blindly. Extract a poster-frame still at a representative timestamp, scaled to the
  same 720px width, and save it alongside the video. No need to ask before compressing —
  just do it and report the before/after size.
- **Never use native `<video controls>` fullscreen or `element.requestFullscreen()`** for
  site video. Two confirmed, separate problems: (1) desktop Safari's native video-fullscreen
  hands off to an AVKit-style player that ignores page CSS and crop-fills 9:16 footage to
  16:9; (2) the standard workaround — fullscreening a wrapper `<div>` instead of the video —
  works everywhere except iPhone Safari, which has never supported the Fullscreen API for
  non-video elements (unresolved as of 2025, no announced fix). Use a plain CSS
  `position: fixed` lightbox overlay instead — hide native controls, add a custom play/expand
  button, and have expand open a shared page-level overlay with the video sized via
  `max-width/max-height: 100%; object-fit: contain`. See the `#vn-lightbox` pattern on
  apps/vandoras.html for the reference implementation — reuse it rather than re-deriving.
  Trade-off (accepted): this isn't true OS-level fullscreen, browser chrome stays visible.

## HTML conventions
- Every page includes the full header and footer (no server-side includes, no JS templating)
- Nav active state: add class `active` to the current page's nav link
- Asset paths are relative: root pages use `assets/`, subpages use `../assets/`
- App Store badges: always use Apple's official badge CDN URL, never a local copy
- External links (App Store, etc): always `target="_blank" rel="noopener"`

## Hamburger nav
JS is inline at the bottom of each page — a minimal toggle for `.nav-hamburger` and
`.site-nav--mobile`. If adding a new page, copy this script block verbatim from any
existing page. Do not refactor into a separate JS file unless Chris explicitly asks.

## What does not exist yet (do not invent)
- Blog or news section
- Contact form (email links only)
- Analytics (add only if Chris asks)

## Source of truth
Brand decisions: `LSS_Brand_Guidelines_v2.2.docx`, on Chris's machine at
`/Users/Chris/Documents/Leslie Street Studio LLC/Branding & Marketing/Guidelines/` —
**outside this repo**, in a sibling folder ("Leslie Street Studio LLC/Branding & Marketing/Guidelines").
Treat it as read-only: never edit it, never deviate from the palette or fonts it defines, and
never attempt to recreate the wordmark — always use the existing SVG/PNG files in
`assets/images/svg/`. The "Brand — LOCKED" section above is a condensed mirror of that
document, current as of v2.2 (April 2026). If a future version of the guidelines conflicts
with anything in this file, flag it to Chris — do not silently resolve it, and don't assume
you can read that path unless it's been shared for the current task.

## Version control
- Remote: `git@github.com:cmac1071/lesliestreetstudio-web.git`, branch `main`. GitHub Pages
  deploys straight from `main` on push — no CI, no build step, no staging branch.
- `.gitignore` currently excludes `CLAUDE.md` itself, `.gitignore`, and `.DS_STORE`. That means
  this file is NOT guaranteed to survive a fresh `git clone` — it only persists because it
  exists on Chris's local disk. If you're picking up this project in a new clone or a new
  machine and this file is missing, that's why. Flag this to Chris — a project-instructions
  file that isn't version-controlled is a real risk for a multi-tool, multi-session workflow;
  he may want it force-added and tracked (`git add -f CLAUDE.md`) rather than staying local-only.
- Don't skip git hooks, force-push, or rewrite history unless Chris explicitly asks.

## Housekeeping Rules
Every time you create new pages, either related to a new app or new structural content
added to lesliestreetstudio.com, be sure to update this CLAUDE.md file in the root of
the project with relevant updates. Also update sitemap.xml to aid search engine crawlers.
If CLAUDE.md ever does get tracked in git (see "Version control" above), commit changes to
it in the same commit as the page changes that prompted them — don't let it drift out of
sync with a separate, forgotten commit.