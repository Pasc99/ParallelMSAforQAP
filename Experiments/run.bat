@echo off
echo Running all on CUDA
FOR %%i IN (4, 8, 16) DO (
echo Running PMSA/CUDA on instance 0, 300 trials, %%i states
START /WAIT PMSAviaCUDA.exe --instance=0 --trial=300 --state=%%i
echo PMSAviaCUDA-%%i on instance 0 is completed. Waiting...
echo;
)
FOR %%i IN (8, 16, 24, 48) DO (
echo Running PMSAviaCUDA on instance 1, 300 trials, %%i states
START /WAIT PMSAviaCUDA.exe --instance=1 --trial=300 --state=%%i
echo PMSAviaCUDA-%%i on instance 1 is completed. Waiting...
echo;
)
FOR %%i IN (8, 16, 32, 64, 512, 1024) DO (
echo Running PMSAviaCUDA on instance 2, 300 trials, %%i states
START /WAIT PMSAviaCUDA.exe --instance=2 --trial=300 --state=%%i
echo PMSAviaCUDA-%%i on instance 2 is completed. Waiting...
echo;
)
FOR %%i IN (8, 16, 32, 128, 256) DO (
echo Running PMSAviaCUDA on instance 3, 300 trials, %%i states
START /WAIT PMSAviaCUDA.exe --instance=3 --trial=300 --state=%%i
echo PMSAviaCUDA-%%i on instance 3 is completed. Waiting...
echo;
)

echo Running all on SA
FOR  /l %%i IN (0,1,3) DO (
echo Running SA on instance %%i, 300 trials, 1 states
START /WAIT SMSA.exe --instance=%%i --trial=300 --state=1
echo SA on instance %%i is completed. Waiting...
echo;
)

echo Running all on SMSA
echo Running SA on instance 0, 300 trials, 4 states
START /WAIT SMSA.exe --instance=0 --trial=300 --state=4
echo SMSA-4 on instance 0 is completed. Waiting...
echo;
FOR  /l %%i IN (0,1,3) DO (
echo Running SA on instance %%i, 300 trials, 8 states
START /WAIT SMSA.exe --instance=%%i --trial=300 --state=8
echo SMSA-8 on instance %%i is completed. Waiting...
echo;
)
FOR  /l %%i IN (0,1,3) DO (
echo Running SA on instance %%i, 300 trials, 16 states
START /WAIT SMSA.exe --instance=%%i --trial=300 --state=16
echo SMSA-16 on instance %%i is completed. Waiting...
echo;
)

echo Running all on OpenMP
FOR %%i IN (4, 8, 16) DO (
echo Running PMSAviaOMP on instance 0, 300 trials, %%i states
START /WAIT PMSAviaOMP.exe --instance=0 --trial=300 --state=%%i
echo PMSAviaOMP-%%i on instance 0 is completed. Waiting...
echo;
)
FOR %%i IN (8, 16, 24, 48) DO (
echo Running PMSAviaOMP on instance 1, 300 trials, %%i states
START /WAIT PMSAviaOMP.exe --instance=1 --trial=300 --state=%%i
echo PMSAviaOMP-%%i on instance 1 is completed. Waiting...
echo;
)
FOR %%i IN (8, 16, 32, 64) DO (
echo Running PMSAviaOMP on instance 2, 300 trials, %%i states
START /WAIT PMSAviaOMP.exe --instance=2 --trial=300 --state=%%i
echo PMSAviaOMP-%%i on instance 2 is completed. Waiting...
echo;
)
FOR %%i IN (8, 16, 32) DO (
echo Running PMSAviaOMP on instance 3, 300 trials, %%i states
START /WAIT PMSAviaOMP.exe --instance=3 --trial=300 --state=%%i
echo PMSAviaOMP-%%i on instance 3 is completed. Waiting...
echo;
)
PAUSE