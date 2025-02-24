@echo off

REM Activate the virtual environment
call D:\Documents\MASC\NLP_EMO\.venv312\Scripts\activate

REM Change directory to the scripts folder
cd D:\Documents\MASC\NLP_EMO\EmoSpeech\Processing

REM Run the Python scripts
python make_old_images.py
python make_old_images.py
python make_old_images.py
python make_old_images.py
python make_old_images.py
python make_old_images.py
python make_old_images.py
python make_old_images.py
python make_old_images.py
python make_old_images.py
python make_old_images.py
python make_old_images.py

REM Deactivate the virtual environment
deactivate
