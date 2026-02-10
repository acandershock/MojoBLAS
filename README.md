# MojoBLAS #

### Description ###

MojoBLAS is a GPU-accelerated implementation of a basic linear algebra 
subprogram (BLAS) library in the programming language Mojo. This library is
an efficient, portable, and easy-to-use BLAS library built directly on Mojo’s native GPU
abstractions. It ensures the technical correctness of Level
1, 2, and 3 BLAS routines based on the original FORTRAN 77 specification, support
for many floating-point precision types in GPU kernels, multi-platform portability across
NVIDIA and AMD hardware, and matches performance in current C++ BLAS libraries.

### Installation Instructions ###

1. Set up your Pixi coding environment.
- If you don't have `pixi` already, you can install it with: 
    ```
    curl -fsSL https://pixi.sh/install.sh | sh
    ```
2. Ensure `pixi` paths are visible.
- On Windows and Linux, close and reopen (ie. restart) the terminal.
- On Mac (or if the above instructions fail), manually add `pixi` to PATH.
    - For zsh shell:
        ```
        echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.zshrc
        source ~/.zshrc
        ```
    - For bash shell:
        ```
        echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
        source ~/.bashrc
        ```
        - For Macs with bash shells, sometimes this command fails to add the `pixi` path to the login shell, which means this path won't be remembered by future shells. If you are running into an issue where new bash shells can't find the pixi executable, try adding it to your bash profile with:
            ```
            echo 'if [ -f ~/.bashrc ]; then . ~/.bashrc; fi' >> ~/.bash_profile
            source ~/.bash_profile
            ```
3. Test `pixi` installation
- If `pixi` has been successfully installed, you will see the `pixi` version return with this test:
    ```
    pixi --version
    ```
4. For Macs, ensure you have XCode downloaded.
- Check your application folder with:
    ```
    ls /Applications/Xcode.app
    ```
- If you see a list of folder contents, you're good to go!
- If not, download XCode from the App Store.
- Then, change the command line tools to point to XCode's toolchain with:
    ```
    sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
    ```
- Accept XCode's license agreement.
    ```
    sudo xcodebuild -license accept
    ```
- Download MetalToolchain:
    ```
    sudo xcodebuild -downloadComponent MetalToolchain
    ```
- And check it was successfully downloaded with:
    ```
    xcrun --find metallib
    ```

### Usage ###

1. Run the Mojo code through Pixi.
- For MacOS, first run:
    ```
    pixi workspace platform add osx-arm64
    ```
    - This will add a pixi.lock file for supported osx-arm64 machines.
    - Then, follow the instructions for Linux-based systems.
- On Linux based systems, run test with:
    ```
    pixi run mojo test-level1.mojo
    ```

### Contact Information ###

For questions about this library, please contact the project leader [Tatiana Melnichenko](tdehoff@vols.utk.edu) 
or [Alexa Andershock](aandersh@vols.utk.edu).

### Acknowledgements ###

This project could not have been completed without contributions from 
Alexa Andershock, Gian Fernandez-Aleman, Tatiana Melnichenko, Jackson Mowry, 
Holden Roaten, and Jonah Weston.