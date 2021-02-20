SET "CV_1=.\x64\Release\CameraCalibration.exe"
SET CAM_NUMBER=4

for /l %%i in ( 1, 1, %CAM_NUMBER% ) do (
    CALL %CV_1% .\tests\cam%%i\calibration.xml
)