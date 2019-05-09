#!/bin/bash

export MAILTO='phongdk@coccoc.com'
export MAILTO_FAILURES='phongdk@coccoc.com'


ensure_dir_exists() {

    if [ ! -d "${1}" ]; then
	mkdir -p "${1}"
    fi
}

export -f ensure_dir_exists

mailTo() {
    if [ $# -ne 2 ] && [ $# -ne 3 ]; then
        echo "Wrong params to mailTo $@. Use with: <email> <subject> [message]"
        exit 1
    else
        echo "Message will be send to $1"
    fi

    address="${1}"
    subject="${2}"
    if [ $# -eq 2 ]; then
        message="${subject}"
    else
        message="${3}"
    fi

    echo "${message}" | mail -s "${subject}" "${address}"
}

export -f mailTo

mailDevelopers() {
	mailTo "${MAILTO}" "${1}" "${2}"
}

export -f mailDevelopers

mailFailure() {
    if [ "$#" -eq 1 ]; then
        mailTo "${MAILTO_FAILURES}" "${1}" "${1}"
    else
        mailTo "${MAILTO_FAILURES}" "${1}" "${2}"
    fi
}

export -f mailFailure

# <code of last command> <message> <not empty to suppress mailing of negative results>
checkResult() {
    status=$1
    message=$2
    if [ $status -ne 0 ]; then
        content="${message} [code ${status}]"
        if [ $# -gt 2 ]; then
            echo "$content"
        else
            echo "$content" | mail -s "${message}" "${MAILTO_FAILURES}"
        fi
        exit ${status}
    fi
}

export -f checkResult

symlink_dest() {
    if [ -h "$1" ]; then
        readlink -f "$1"
    else
        checkResult 1 "File $1 is not symlink."
    fi
}

export -f symlink_dest

# <code of last command> <message> <not empty to suppress mailing of negative results>
exitWithError() {
    if [ $# -eq 1 ]; then
        message=$1
    else
        message="Wrong args to exitWithError: $@"
    fi
    echo ${message}
    echo "${message}" | mail "${MAILTO}" -s "${message}"
    exit 1
}

export -f exitWithError

get_revision() {
    local rev=$(get_revision_quietly "$1")
    if [ -z "${rev}" ]; then
        exitWithError "Can't get revision at ${1}"
    else
        echo "${rev}"
    fi
}

export -f get_revision

get_revision_quietly() {
    if [ $# -ne 1 ]; then
        exitWithError "Use $0 with args: <dir>"
    fi

    if [ -d "${1}" ]; then
	local cur=${PWD}

	revision=$(jar xf ${1}/build/libs/itim-code.jar META-INF/MANIFEST.MF && cat META-INF/MANIFEST.MF | grep Implementation-Version | awk '{print $2}' && rm -rf META-INF)

	if [ -z "${revision}" ]; then
            echo ""
	else
            echo "${revision}"
	fi

	cd ${cur}
    else
	echo ""
    fi
}

export -f get_revision_quietly

assert_revision() {
    if [ "$#" -ne 2 ]; then
        exitWithError "Use with args: <dir> <revision>"
    fi

    rev=$(get_revision $1)
    if [ "${rev}" -ne "${2}" ]; then
        exitWithError "Wrong revision ${rev} (${2}) at ${1}"
    fi
    echo "$1 with revision $2"
}

export -f assert_revision

ensureDirExists() {
    if [ ! -d $1 ]; then
        mkdir -p ${1}
    fi
}

export -f ensureDirExists

assertDirExists() {
    if [ ! -d "$1" ]; then
	echo "Dir ${1} is not exist."
        exit 1
    fi
}

export -f assertDirExists

assertFileExists() {
    if [ ! -f "${1}" ]; then
	echo "File ${1} is not exists."
        checkResult 1 "File ${1} is not found"
    fi
}

export -f assertFileExists

ensureFileEmpty() {
	if [ -f ${1} ]; then
		rm ${1}
	fi

	touch ${1}
}

maxFromIntList() {
    prev=""
    for val in $1; do
        if [ -z ${prev} ] || [ ${prev} -lt ${val} ]; then
            prev=${val};
        fi
    done

    echo ${prev}
}

execBySSH() {
    if [ $# != 2 ]; then
        echo "Not enough params for execBySSH. Use with args: <server> <command>"
        exit 1
    fi

    output=`ssh "$1" "$2" </dev/null`
    exitCode=$?
    if [ ${exitCode} -ne 0 ]; then
        sleep 10
        output=$(ssh "$1" "$2" </dev/null)
        exitCode=$?
    fi

    echo "$output"
    return ${exitCode}
}

export -f execBySSH

time_echo() {
    echo `eval date +%Y-%m-%d_%H.%M.%S`" $1"
}

export -f time_echo

print_time() {
	echo `eval date +%Y-%m-%d_%H.%M.%S`
}

# use with args: <substring in proc name> <timeout in seconds>
# return: 0 - proc finished, 1 - not finished
wait_proc_finished() {

	if [ $# -ne 2 ]; then
        echo "Use wait_proc_finished() with args: <proc substring> <timeout>"
        exit 1
	fi

	local proc_pattern="${1}"
    local timeout="${2}"
    local start=`date +%s`
    local proc=$(ps aux | grep "${proc_pattern}" | grep -v grep | wc -l)

    while [ "${proc}" -ne 0 ]; do
        local current=`date +%s`
        local spent=$(( current - start ))

        if [ "${timeout}" -ne -1 ] && [ "${spent}" -gt "${timeout}" ]; then
			echo "1"
			return
        fi

        sleep 5
        proc=$(ps aux | grep "${proc_pattern}" | grep -v grep | wc -l)
    done

	echo "0"
}

export -f wait_proc_finished

stop_process() {
    local proc_pattern=$1
    local timeout=$2

    local proc=`ps aux | grep ${proc_pattern} | grep -v grep | head -n 1 | awk '{print $2}'`

    if [ ! -z "${proc}" ]; then
        kill -15 ${proc}
        wait_proc_finished "${proc_pattern}" "${timeout}"
    fi
}

export -f stop_process

cut_www_from_host() {
    if [ $# != 1 ]; then
        exitWithError "Use $0 with: <hostname>"
    fi

    echo $1 | awk -F'.' '{
        parts=NF;
        if (parts >= 3 && $1 ~ "^www[0-9]*$") {
             host=$2;
             for (i=3 ; i <= NF ; i++ ) {
                 host=host"."$i
             }
        } else {
            host=$0
        }
        print host
    }'
}

export -f cut_www_from_host

warnCountOfFilesInDir() {
    dir=$1
    count=$2
    if [ ! -d "${dir}" ]; then
        msg="There is no ${dir}"
        mailDevelopers "${msg}" "${msg}"
        return
    fi

    counted=`ls $1 | wc -l`
    if [ "${counted}" -gt $2 ]; then
        msg="Too many files in ${dir}. ${counted} instead of max ${count}"
        mailDevelopers "${msg}" "${msg}"
        return
    fi
}

export -f warnCountOfFilesInDir

get_debian_major_version() {
    awk '{if ($1=="Debian") print $3}' /etc/issue | head -n 1 | cut -d '.' -f1
}

export -f get_debian_major_version

# <folder> <count of files>
assertCountOfFiles() {
    if [ $# -ne 2 ]; then
        echo "Use with args: <folder> <count of files>"
        return 1
    fi
    folder=$1
    expected=$2
    count=`ls ${folder} | wc -l`
    if [ "$count" -eq "$expected" ]; then
        echo "$folder contains $count files as expected"
    else
        message="Found $count files in $folder instead of $expected."
        echo "$message" | mail -s "${message}" $MAILTO
        exit -1
    fi
}

assertDefined() {
    local name="${1}"
    if [ -z "${name}" ]; then
        time_echo "variable name isn't defined."
        exit 1
    fi

    local value="${!name}"
    if [ -z "${value}" ]; then
	time_echo "variable ${name} is not defined."
	exit 1
    fi
}

export -f assertDefined

getTimestamp() {
    date +%Y-%m-%d_%H.%M.%S
}

export -f getTimestamp

checkJavaExitCode() {
    if [ "$#" -eq 2 ]; then
        processName="$1"
        exitCode="$2"
        if [ "${exitCode}" -eq 137 ]; then
            checkResult ${exitCode} "${processName} killed by SIGKILL"
        elif [ "${exitCode}" -eq 143 ]; then
            echo "${processName} stopped by SIGTERM"
        else
            checkResult "${exitCode}" "${processName} failed with code ${exitCode}"
        fi
    fi
}

export -f checkJavaExitCode

# <path to a diretory to create>
createDirIfMissed() {
    path=$1
    if [ -e "$path" ]; then
        if [ ! -d "$path" ]; then
            echo "$path is not a directory"
        fi
    else
        mkdir -p "$path"
        checkResult $? "Can't create directory $path"
    fi
}

export -f createDirIfMissed

count_java_proc() { # tested
    local mask=$1
    local proc=`ps aux | grep "${mask}" | grep java | grep -v grep | wc -l`
    checkResult $? "Failed ${mode} detection"
    echo $proc;
}

export -f count_java_proc

kill_java_proc() {
    cnt=$(count_java_proc "${1}")

    if [ "${cnt}" -gt 0 ]; then
	ps aux | grep java | grep "${1}" | grep -v grep | awk '{print $2}' | xargs kill

	while [ $(count_java_proc "${1}") -gt 0 ]; do
	    sleep 1
	done
    fi
}

export -f kill_java_proc

count_files() { # tested
    find $1 -type f | grep "$2" | wc -l
}

export -f count_files

execution_time() {
    local etime=$(ps axo "%t %a" | grep "${1}" | grep -v grep | grep java | awk '{print $1}' | head -n 1)
    if [ -z "${etime}" ]; then
        echo -1
        return
    fi

    local seconds=$(echo "${etime}" | tr '-' ' ' | tr ':' ' ' | awk '
         {
             if (NF == 1) sum=$1
             if (NF == 2) sum=60 * $1 + $2
             else if(NF == 3) sum=3600 * $1 + 60 * $2 + $3
             else if(NF == 4) sum=24 * 3600 * $1 + 3600 * $2 + 60 * $3 + $4
             else sum=-1
         }
         END{ print sum }
    ')

    if [ "${seconds}" -eq -1 ]; then
        time_echo "Can't parse elapsed time for ${2} at ${1}"
        exit 2
    else
        echo ${seconds}
    fi
}

export -f execution_time

# Normally should be no old files in directory for data exchange,
# but if we stopped processes by hands. Some temporary files may remain.
cleanup_directory() {
    dir=$1
    warn_days=$2
    remove_days=$3
    pattern=$4

    remove=$(find "${dir}/" -mindepth 1 -maxdepth 1 -type f -mtime "+${remove_days}" | grep "${pattern}" | wc -l)
    if [ "${remove}" -gt 0 ]; then
        find "${dir}/" -mindepth 1 -maxdepth 1 -type f -mtime "+${remove_days}" | grep "${pattern}" | xargs rm
        checkResult $? "Failed removal of obsolete files from ${dir}"
        msg="WARNING ${remove} obsolete files in ${HOSTNAME} ${dir} were removed."
        mailFailure "${msg}"
    fi

    warn=$(find "${dir}/" -mindepth 1 -maxdepth 1 -type f -mtime "+${warn_days}" | grep "${pattern}" | wc -l)
    if [ "${warn}" -gt 0 ]; then
        msg="WARNING ${warn} obsolete files in ${HOSTNAME} ${dir}"
        mailFailure "${msg}" "${msg}. They will be removed automatically after become ${remove_days} days old."
    fi
}

export -f cleanup_directory

logname_time() {
    date +%Y-%m-%d_%H.%M.%S
}

export -f logname_time

call_blocking() {
    local url="${1}"
    local doneMsg="${2}"
    local errMsg="${3}"

    local done=$(curl -s "${url}")

	if [ "${done}" == "true" ]; then
	    time_echo "# ${doneMsg}"
	else
        checkResult 1 "${errMsg}"
	fi
}

export -f call_blocking

function check_empty_var {
    if [[ -z "${1}" ]]; then
        exitWithError "ERROR ${2}"
    fi
}

export -f check_empty_var

push_to_slack() {
    local address="https://hooks.slack.com/services/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"
    local text="`echo "${1}" | sed ':a;N;$!ba;s/\n/\\\\n/g' | sed 's/"/\\\\"/g'`"
    # local msg="payload={\"channel\": \"#demo-noti\", \"username\": \"notificator\", \"text\": \"${text}\", \"icon_emoji\": \":flag-vn:\"}"
    # curl -X POST -H 'Content-type: application/json' --data-urlencode "${msg}" "${address}"
    curl -X POST -H 'Content-type: application/json' --data "{\"text\": \"${text}\"}" ${address}
}

export -f push_to_slack

checkResult_slack() {
    status=$1
    message=$2
    if [ ${status} -eq 0 ]; then
        echo "${message} ${date} : PASS"
    else
        echo "${message} ${date} : ERROR"
        #slack-message
        push_to_slack "${message} ${date} : ERROR"
        exit ${status}
    fi
}

export -f checkResult_slack

