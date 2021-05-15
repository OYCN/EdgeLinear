
add_executable(
    EDbaseline

    EDProcess.cpp
    EDmain.cpp
)

add_executable(
    EDbaseline-unit

    EDProcess.cpp
    EDmain.cpp
)

target_compile_definitions(EDbaseline-unit PRIVATE UNIT_RESULT)

add_executable(
    EDDPbaseline

    EDProcess.cpp
    EDDPmain.cpp
)

add_executable(
    EDLSbaseline

    EDProcess.cpp
    EDLSmain.cpp
)