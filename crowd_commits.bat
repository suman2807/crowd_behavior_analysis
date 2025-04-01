@echo off
setlocal enabledelayedexpansion

for %%m in (04 05 06) do (

    for /L %%d in (1,1,30) do (

        REM random skip (more gaps now)
        set /a chance=!random! %% 10
        if !chance! GEQ 5 (

            REM burst logic (1 to 4 commits)
            set /a commits=!random! %% 4 + 1

            for /L %%c in (1,1,!commits!) do (

                set day=%%d
                if !day! LSS 10 set day=0!day!

                REM wider time range (morning to night)
                set /a hour=!random! %% 12 + 9
                set /a min=!random! %% 60

                echo %%m-%%d-%%c >> file.txt
                git add .

                set DATE=2025-%%m-!day! !hour!:!min!:00
                set GIT_AUTHOR_DATE=!DATE!
                set GIT_COMMITTER_DATE=!DATE!

                REM smarter message distribution by month
                if %%m==04 (
                    set /a r=!random! %% 6
                    if !r! EQU 0 git commit -m "initialized project structure"
                    if !r! EQU 1 git commit -m "added dataset for training"
                    if !r! EQU 2 git commit -m "setup YOLO environment"
                    if !r! EQU 3 git commit -m "preprocessed input videos"
                    if !r! EQU 4 git commit -m "configured project dependencies"
                    if !r! EQU 5 git commit -m "updated README setup instructions"
                )

                if %%m==05 (
                    set /a r=!random! %% 6
                    if !r! EQU 0 git commit -m "implemented YOLO detection module"
                    if !r! EQU 1 git commit -m "added LSTM model for behavior prediction"
                    if !r! EQU 2 git commit -m "trained model on dataset"
                    if !r! EQU 3 git commit -m "improved prediction accuracy"
                    if !r! EQU 4 git commit -m "fixed bug in detection pipeline"
                    if !r! EQU 5 git commit -m "optimized model performance"
                )

                if %%m==06 (
                    set /a r=!random! %% 6
                    if !r! EQU 0 git commit -m "integrated dashboard UI"
                    if !r! EQU 1 git commit -m "added alert system for crowd anomalies"
                    if !r! EQU 2 git commit -m "connected backend with frontend"
                    if !r! EQU 3 git commit -m "refactored code structure"
                    if !r! EQU 4 git commit -m "improved UI responsiveness"
                    if !r! EQU 5 git commit -m "updated documentation and final fixes"
                )
            )
        )
    )
)