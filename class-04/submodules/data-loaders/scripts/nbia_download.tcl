#!/usr/bin/expect/

# =============================================================================
# nbia_download.tcl
#
# Description:
# Convenience script to download a dataset with the NBIA Data Retriever
# command line interface. The NBIA Data Retriever is used to download data from
# The Cancer Imaging Archive (TCIA). This script is intended for use inside the
# devcontainer that accompanies this repository.
#
# Arguments:
#    Argument 1 - specify the type of download: valid values are DOWNLOAD (use
#    for a new download), REDOWNLAD (for redownloading a previous download),
#    and RESUME (for resuming an unfinished download). Note that REDOWNLOAD
#    and RESUME will only work for versions of the NBIA installer that support
#    resuming downloads. For other versions of the installer, just use
#    DOWNLOAD.
#
#    Argument 2 - specify the timeout: specify the maximum time to wait for
#    the download to complete in seconds (e.g., 30). Use -1 for no timeout.
#    No timeout is the preferred setting for long downloads.
#
#    Argument 3 - specify the manifest file: This is the full unix path to
#    the manifest file, e.g., /path/to/manifest.tcia.
#
#    Argument 4 - specify the location to store the download files: This is the
#    full path to the download location, e.g., /path/to/downloads.
#
# Dependencies:
#
# This script assumes that expect is already installed in the environment.
#
# Usage:
#
# expect nbia_download.tcl (DOWNLOAD | REDOWNLOAD | RESUME) <timeout> <manifest_filename> <download_location>
#
# Examples:
#
# Use the DOWNLOAD argument to start a new download.
#
# expect nbia_download.tcl DOWNLOAD -1 \
#    /workspaces/data-loaders/nbia_manifest_files/small_test_manifest/manifest-*.tcia \
#    /workspaces/data-loaders/downloads/nbia
#
# Use the REDOWNLOAD argument to completely redownload a previously downloaded
# dataset (e.g., if the previously downloaded dataset is corrupted). This will
# redownload all files in the manifest, even if some files are already present
# in the specified downloads folder.
#
# expect nbia_download.tcl REDOWNLOAD -1 \
#    /workspaces/data-loaders/nbia_manifest_files/small_test_manifest/manifest-*.tcia \
#    /workspaces/data-loaders/downloads/nbia/
#
# Use the RESUME argument to resume a previous download. This will only download
# files not present in the previous download.
#
# expect nbia_download.tcl RESUME -1 \
#    /workspaces/data-loaders/nbia_manifest_files/small_test_manifest/manifest-*.tcia \
#    /workspaces/data-loaders/downloads/nbia/
#
# =============================================================================

# Global variables
set SCRIPT_NAME "nbia_download.tcl"
set VERSION "0.0.1"
set DOWNLOAD_ARG "DOWNLOAD"
set REDOWNLOAD_ARG "REDOWNLOAD"
set RESUME_ARG "RESUME"
set CORRECT_ARGC 4

# Documentation string for usage explanations
set DOCUMENTATION {
Convenience script to download a dataset with the NBIA Data Retriever
command line interface. The NBIA Data Retriever is used to download data from
The Cancer Imaging Archive (TCIA). This script is intended for use inside the
devcontainer that accompanies this repository.

Dependencies:

This script assumes that expect is already installed in the environment.

Arguments:
   Argument 1 - specify the type of download: valid values are DOWNLOAD (use
   for a new download), REDOWNLAD (for redownloading a previous download),
   and RESUME (for resuming an unfinished download). Note that REDOWNLOAD
   and RESUME will only work for versions of the NBIA installer that support
   resuming downloads. For other versions of the installer, just use
   DOWNLOAD.

   Argument 2 - specify the timeout: specify the maximum time to wait for
   the download to complete in seconds (e.g., 30). Use -1 for no timeout.
   No timeout is the preferred setting for long downloads.

   Argument 3 - specify the manifest file: This is the full unix path to
   the manifest file, e.g., /path/to/manifest.tcia.

   Argument 4 - specify the location to store the download files: This is the
   full path to the download location, e.g., /path/to/downloads.

Examples:

Use the DOWNLOAD argument to start a new download.

expect nbia_download.tcl DOWNLOAD -1\
    /workspaces/data-loaders/nbia_manifest_files/small_test_manifest/manifest-*.tcia\
    /workspaces/data-loaders/downloads/nbia/

Use the REDOWNLOAD argument to completely redownload a previously downloaded
dataset (e.g., if the previously downloaded dataset is corrupted). This will
redownload all files in the manifest, even if some files are already present
in the specified downloads folder.

expect nbia_download.tcl REDOWNLOAD -1\
/workspaces/data-loaders/nbia_manifest_files/small_test_manifest/manifest-*.tcia\
/workspaces/data-loaders/downloads/nbia/

Use the RESUME argument to resume a previous download. This will only download
files not present in the previous download.

expect nbia_download.tcl RESUME -1\
/workspaces/data-loaders/nbia_manifest_files/small_test_manifest/manifest-*.tcia\
/workspaces/data-loaders/downloads/nbia/
}

##
# Print the usage message.
#
# Parameters:
#     None
#
# Return Value:
#     None
#
proc print_usage {} {
    global SCRIPT_NAME
    global DOWNLOAD_ARG
    global REDOWNLOAD_ARG
    global RESUME_ARG

    set usage_string ""
    append usage_string "Usage: expect $SCRIPT_NAME"
    append usage_string " ($DOWNLOAD_ARG | $REDOWNLOAD_ARG | $RESUME_ARG)"
    append usage_string " (-1 | <timeout value>)"
    append usage_string " <manifest_filename> <download_location>"

    puts "$usage_string"
    puts "Example: expect scripts/nbia_download.tcl DOWNLOAD -1 /workspaces/data-loaders/nbia_manifest_files/small_test_manifest/manifest-*.tcia /workspaces/data-loaders/downloads/nbia/"
    puts "Help: expect scripts/nbia_download.tcl --help"
    puts "Version: expect scripts/nbia_download.tcl --version"
}

##
# Print the version message.
#
# Parameters:
#     None
#
# Return Value:
#     None
#
proc print_version {} {
    global SCRIPT_NAME
    global VERSION
    puts "$SCRIPT_NAME: version $VERSION"
}

##
# Print the help message.
#
# Parameters:
#     None
#
# Return Value:
#     None
#
proc print_help {} {
    global DOCUMENTATION
    puts ""
    print_version
    puts "\n==========="
    puts "Quick help:"
    puts "===========\n"
    print_usage
    puts "\n==============="
    puts "Detailed usage:"
    puts "==============="
    puts $DOCUMENTATION
}

##
# Check for optional arguments like --version and --help.
#
# Parameters:
#     argv - The argument vector, directly from STDIN.
#
# Return Value:
#     None
#
proc check_for_options {argv} {
    foreach arg $argv {
        if {[string match "--help" $arg]} {
            print_help
            exit 0
        } elseif {[string match "--version" $arg]} {
            print_version
            exit 0
        } else {
            puts " (nbia_download): Invalid input argument."
            print_usage
            exit -1
        }
    }
}

##
# Checks for the correct argument count.
#
# Parameters:
#     None
#
# Return Value:
#     None
#
proc check_argument_count {argc} {
    global CORRECT_ARGC
    if { $argc != $CORRECT_ARGC } {
        puts " (nbia_download): Not enough input arguments."
        print_usage
        exit 1
    }
}

proc extract_valid_input_arguments {argv argc} {
    if { $argc == 1 } {
        check_for_options $argv
    }
    check_argument_count $argc
    set download_mode [lindex $argv 0]
    set manifest_file [lindex $argv 2]
    set download_location [lindex $argv 3]
    set download_mode [string toupper $download_mode]
    return [list $download_mode $manifest_file $download_location]
}

set validated_arguments [extract_valid_input_arguments $argv $argc]
set download_mode [lindex $validated_arguments 0]
set manifest_file [lindex $validated_arguments 1]
set download_location [lindex $validated_arguments 2]

set timeout [lindex $argv 1]

spawn /opt/nbia-data-retriever/nbia-data-retriever \
    --cli "$manifest_file" \
    -d "$download_location" \
    -v -F

expect "Do you agree with the Data Usage Agreement? (Y/N)"
send "Y\n"
switch $download_mode {
    "DOWNLOAD" {
        # Since this is a first download, there are no other interactions to
        # be expected.
        puts " (nbia_download): Downloading new files..."
    }
    "REDOWNLOAD" {
        # Since this is a complete redownload, we need all the files.
        puts " (nbia_download): Redownloading files..."
        expect "Enter A for downloading all, M for downloading missing series. E for exiting the program:"
        send "A\n"
    }
    "RESUME" {
        # Since this is a resumed download, we only need to download the
        # missing series.
        puts " (nbia_download): Resuming previous download..."
        expect "Enter A for downloading all, M for downloading missing series. E for exiting the program:"
        send "M\n"
    }
    default {
        puts " (nbia_download): Unknown argument to $SCRIPT_NAME: $download_mode"
        print_usage
        exit 1
    }
}

puts " (nbia_download): Complete!"

expect {
    eof {
        puts " (nbia_download): Reached end of file. Exiting normally."
    } timeout {
        puts "Connection timed out. Download was not successful."
    }
}
