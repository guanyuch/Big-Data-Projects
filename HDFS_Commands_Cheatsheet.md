## Hadoop Distributed File System (HDFS) Commands Cheat Sheet

#### Most of the commands in FS shell behave like corresponding Unix commands. This cheat sheet is for operating file management in Hadoop Distributed File System (HDFS)

$ hdfs dfs -cat\
Prints to the terminal the contents of the file

$ hdfs dfs -count\
Counts the number of directories, files and bytes under the path

$ hdfs dfs -cp\
Copies a file to a different file/directory

$ hdfs dfs -get\
Downloads a file from HDFS to your local machine

$ hdfs dfs -ls\
Lists files in the directory

$ hdfs dfs -mkdir\
Creates parent directories along the path

$ hdfs dfs -mv\
Moves a file to a different file/directory

$ hdfs dfs -put\
Uploads a file from your local machine to a specific directory on HDFS

$ hdfs dfs -rm\
Deletes a file (sends it to trash)\

$ hdfs dfs -rm -r\
Recursively deletes a directory and it's contents

$ hdfs dfs -stat\
Returns the stat information on the path

$ hdfs dfs -tail\
Displays last kilobyte of the file

$ hdfs dfs -text\
Prints the contents of a file to the termial, decompressing if necessary
#### If you need detailed instruction and more commands, please refer to the link: [Apache Hadoop 2.4.1 -File System Shell Guide](https://hadoop.apache.org/docs/r2.4.1/hadoop-project-dist/hadoop-common/FileSystemShell.html)