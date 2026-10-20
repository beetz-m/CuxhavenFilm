# Cuxhaven film

A film about the Cuxhaven nacelle site research twin (`~/Documents/CuxhavenSiteTwin`). It runs from the site and the production lines through logistics and load-out to the offshore installation, the rotor, and robots servicing the turbine.

The twin is used **read-only**: this project reads its renders, clean frame sequences, stills, register and service scene by path, and writes only here.

## What is in the film

- **Existing material.** The twin's process renders come from their clean 12 fps frames (no caption bar) and are blended to 30 fps and graded per shot. The film also uses its physics episodes (station G5, station B2, the filter change) and its path-traced stills, shown with slow camera moves.
- **New material.** `film/render_service_anim.py` renders seven shots of the robot service visit M1 (Isaac Sim 6.2 headless, RTX real-time with DLAA). The approach, the inspection and contact drones on the blades, the ANYmal hoisted into the nacelle, ANYmal and Spot at work, and the cargo drone. Shots go to `build/service/`.
- **Titles and captions.** `film/titles.py` writes the titles and captions. Every number is read from the twin's register and shows the register's status label (SOURCED, PROXY, ILLUSTRATIVE, …).
- **Score.** `film/soundtrack.py` generates a procedural score (no third-party music).
- **Not used.** The Siemens Gamesa reference video in the twin's `reference/` folder is their copyrighted material, kept only for comparison. It is not in the film.

## Build

    E=~/Documents/LabAutomationTwin/env_cad/bin/python
    I=~/Documents/LabAutomationTwin/env_isaacsim/bin/python
    OMNI_KIT_ACCEPT_EULA=YES $I film/render_service_anim.py   # one GPU job: stop the Isaac GUI first
    $E film/assemble.py --preview                            # 960x540 check, ~3 min
    $E film/assemble.py                                      # build/cuxhaven_film.mp4, 1920x1080, 30 fps

The edit (shots, order, timing, titles) is `film/edl.py`.

## Disclaimer

This is a research reconstruction from public sources. It is not Siemens Gamesa material and uses no internal data.
