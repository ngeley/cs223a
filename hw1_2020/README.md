# CS223a: Introduction to Robotics

## Installation

1. This code officially supports Mac and Ubuntu, although it is possible to run it on Windows using the Windows Subsystem for Linux.

   ### Mac
   
   1. Install Brew (https://brew.sh)
   2. Run ```xcode-select --install```in the terminal to install MacOS command line tools
   
   ### Ubuntu
   
   1. Go to step 2!
   
   ### Windows
   
   1. Install Windows Subsystem for Linux (https://docs.microsoft.com/en-us/windows/wsl/install-win10)
   2. Launch an Ubuntu Bash terminal
   3. Navigate to the folder where you downloaded the code. For example, if you downloaded it to ```C:\Users\Student\Documents\cs223a\hw1``` in the Windows filesystem, you can access it from Ubuntu by navigating to ```/mnt/c/Users/Student/Documents/cs223a/hw1```
   
      ```
      cd /mnt/c/Users/Student/Documents/cs223a/hw1
      ```

2. In case you're unfamiliar with the terminal, here are a few commands you might find helpful for this class:
   
      ```
      cd dir1/dir2   # Navigate to the directory "dir1/dir2" (directories separated by "/")
      cd ..          # Navigate to the parent directory
      ls             # List the files in the current directory
      ./executable   # Execute the file "executable"
      ```
      
      Whenever you're typing a long path name, you can press the ```<TAB>``` key to autocomplete and avoid having to type the entire path.

3. Install Python 3, pip, and virtualenv. You can do this automatically with the following script, or use your own preferred method.

   ### Automatic

   ```
   ./setup_python.sh
   ```

   ### Manual

   ```
   pip install virtualenv           # Install virtualenv if you don't already have it
   virtualenv -p python3 env        # Create a new virtual environment for python3 in the folder "./env"
   source env/bin/activate          # Activate the virtual environment
   pip install -r requirements.txt  # Install the required Python packages in the virtual environment
   ```

## Usage

1. Activate the Python virtual environment from the top directory. You will have to do this every time you open a new terminal. The virtual environment ensures that the CS223a code and its dependencies won't affect other Python projects on your system.

   ```
   source env/bin/activate
   ```
   
   If the environment is activated, you should see something like ```(env)``` at the beginning of your terminal input line.
  
2. The code you need to implement will be in the ```cs223a``` folder.

3. At the bottom of each file where you need to implement code is a main function that contains tests of the functions you write. Running the file by itself while in the environment will execute these tests (in this case, running "python rotations.py" while you are inside the cs223a folder). The file includes some very basic common sense tests, but passing the given tests does not mean your function works properly.
