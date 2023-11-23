#!/bin/bash

# Replace with your actual remote server username and hostname
remote_user="jkrupa"
remote_host="submit07.mit.edu"
remote_path="/work/tier3/jkrupa/cl/deepjet-geometric/examples/nov23"

# Replace with your destination path
local_path="./"

# Fetch the list of directories containing 'COMPLETED' into an array
IFS=$'\n' read -r -d '' -a dirs < <(ssh $remote_user@$remote_host "find $remote_path -type d -name '*COMPLETED*'" && printf '\0')

# Iterate over the directories
for dir in "${dirs[@]}"; do

    local_dir="$local_path/$(basename "$dir")"

    # Skip if the local directory already exists
    if [ -d "$local_dir" ]; then
        echo "Directory $local_dir already exists, skipping..."
        continue
    fi

    # Check if the file exists on the remote server
    if ssh $remote_user@$remote_host "[ -f $dir/FT_best-epoch.pt ]"; then
        # Create local directory
        mkdir -p "$local_dir"
        
        # Copy the file
        scp "$remote_user@$remote_host:$dir/FT_best-epoch.pt" "$local_dir/"
    fi
done

