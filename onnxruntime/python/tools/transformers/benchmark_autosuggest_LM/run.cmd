::#####################
::#### DO NOT EDIT ####
::#####################

setlocal
set CURRENT_PATH=%~dp0
set CURRENT_PATH=%CURRENT_PATH:~0,-1%

if "%1"=="--gpu" (
   shift /1
   @echo GPU mode detected! This setting is now deprecated.
)

set argC=0
for %%x in (%*) do Set /A argC+=1

IF "%1%" == "http" (GOTO http)
IF "%1%" == "grpc" (GOTO grpc)
IF "%1%" == "udp" (GOTO udp)
IF "%1%" == "offline" (GOTO offline)
IF "%1%" == "offline_profile" (GOTO offline_profile)
GOTO usage

:http
python %CURRENT_PATH%\model\main.py http
GOTO end

:grpc
python %CURRENT_PATH%\model\main.py grpc
GOTO end

:udp
python %CURRENT_PATH%\model\main.py udp
GOTO end

:offline
IF %argC% == 3 (
    python %CURRENT_PATH%\model\main.py offline %2 %3
    GOTO end
)
IF %argC% == 1 (
    python %CURRENT_PATH%\model\main.py offline %_InputFilePath_% %_OutputFilePath_%
    GOTO end
)
GOTO usage

:offline_profile
IF NOT %argC% == 4 (GOTO usage)
python -m cProfile -o %4.stats %CURRENT_PATH%\model\main.py offline %2 %3
gprof2dot -f pstats %4.stats | dot -Tpng -o %4.png
GOTO end

:usage
echo "Usage:"
echo "run.cmd http"
echo "run.cmd grpc"
echo "run.cmd udp"
echo "run.cmd offline"
echo "run.cmd offline inputfile outputfile"
echo "run.cmd offline_profile inputfile outputfile profilename"
GOTO end

:end
