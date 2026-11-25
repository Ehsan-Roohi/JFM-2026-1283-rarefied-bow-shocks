# Reproducing the supplementary recordings and Figure S11

Use Python 3.11 or later with the packages in requirements.txt. The two movie scripts use FFmpeg with the libx264 encoder. The default data paths assume this directory is code/publication_movies in the repository. Restore the assets listed in the repository manifest before rendering. Choose a fresh output directory for a fresh dataset.

The formation script reads the sixteen high-particle numerical field arrays from data/physics/video_runs/npz_hi, together with cylinder.surf and plate.surf. It produces movies 5–12 and the scale and QA records. From the repository root run:

    python code/publication_movies/render_formation.py --runs data/physics/video_runs --output reproduced

To regenerate Figure S11 from the late-time windows and the archived time-averaged velocity fields in uv_hi:

    python code/publication_movies/render_s11.py --runs data/physics/video_runs --output reproduced/figures/Aniso_contours.pdf

The saved samples supply density, translational and rotational temperature, and the signed normal-stress ratio. The formation and Figure S11 scripts test pressure and realizability before display. The shared movie scale for the stress ratio spans −3 to +3. The static late-window Figure S11 uses a common −2 to +2 scale covering its displayed values.

For movies 1–4, use code/physics/film_runs/make_molecule_film.py with the restored molecule records for argon and nitrogen at Kn = 0.01 and 0.1. Keep the four 1920×1080 outputs named molecules_Ar_Kn0p01.mp4, molecules_Ar_Kn0p1.mp4, molecules_N2_Kn0p01.mp4 and molecules_N2_Kn0p1.mp4 in one directory. Then run:

    python code/publication_movies/render_particles.py --input PATH_TO_THOSE_FOUR_FILMS --output reproduced

The publication pass changes the lower legend and retains the original plotted tracks, counter and clock. The published Windows render used Segoe UI. Systems without it use DejaVu Sans for the legend or accept explicit TrueType fonts. FFmpeg and font versions can change encoded bytes. The scientific data and film numbers are independent of the codec and font files.
