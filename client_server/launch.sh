#!/bin/bash
#how to launch : chmod u+x launch.sh
#bash launch.sh sync/async n  (synchrone or asynchrone, n workers)
#change path  to match local directory
if [ "$1" = "sync" ]
then
  echo "sync mode"
  if [ "$2" -gt 0 ] && [ "$2" -lt 10 ]
  then
   #gnome-terminal --working-directory=/home/cours/epfl/system_for_data_science/project/client_server
    python server.py &
    for i in `seq 1 $2`;
      do
      gnome-terminal --working-directory=/home/cours/epfl/system_for_data_science/project client_server -e "python client.py"
    done
  else
    echo "second argument (#workers) must be between 2 and 10"
  fi
elif [ "$1" = "async" ]
then
  echo "async mode"
else
  echo "first argument must be 'sync' or 'async'"
  exit 1
fi
