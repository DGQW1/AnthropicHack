#NOT USED
import os
import glob
import subprocess
import platform
from pptx import Presentation

def main():
    # Path to the slides directory (absolute path)
    slides_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '270slides'))
    
    # Get all .pptx files in the slides directory
    pptx_files = glob.glob(os.path.join(slides_dir, '*.pptx'))
    
    print(f"Found {len(pptx_files)} PowerPoint files to convert.")
    
    # Check if we're on MacOS
    is_macos = platform.system() == 'Darwin'
    
    for pptx_file in pptx_files:
        base_name = os.path.splitext(os.path.basename(pptx_file))[0]
        pdf_file = os.path.splitext(pptx_file)[0] + '.pdf'
        
        print(f"Converting: {os.path.basename(pptx_file)} -> {os.path.basename(pdf_file)}")
        
        try:
            if is_macos:
                # On MacOS we can use the built-in automator/Quartz filters for conversion
                # This requires that PowerPoint is installed on the machine
                script = f'''
                tell application "Microsoft PowerPoint"
                    open "{pptx_file}"
                    set pres to active presentation
                    save pres in "{pdf_file}" as save as PDF
                    close pres saving no
                end tell
                '''
                
                # Save script to a temporary file
                script_file = os.path.join(os.path.dirname(pptx_file), f"temp_convert_{base_name}.scpt")
                with open(script_file, "w") as f:
                    f.write(script)
                
                # Run the AppleScript
                result = subprocess.run(["osascript", script_file], capture_output=True, text=True)
                
                # Remove the temporary script file
                os.remove(script_file)
                
                if result.returncode == 0:
                    print(f"Successfully converted {os.path.basename(pptx_file)}")
                else:
                    print(f"Error converting {os.path.basename(pptx_file)}: {result.stderr}")
                    print("Make sure Microsoft PowerPoint is installed on your Mac.")
            else:
                # For non-MacOS systems, print instructions
                print(f"Could not automatically convert {os.path.basename(pptx_file)} on this platform.")
                print("For automatic conversion on non-MacOS systems, install LibreOffice.")
                
        except Exception as e:
            print(f"Error converting {os.path.basename(pptx_file)}: {str(e)}")
    
    print("\nConversion complete.")
    print("\nIf conversions failed, you can try these alternatives:")
    print("1. Install LibreOffice and run: ")
    print("   soffice --headless --convert-to pdf --outdir /path/to/output /path/to/presentation.pptx")
    print("2. Use an online conversion service")
    print("3. Open the files in PowerPoint and use 'Save As PDF' option")

if __name__ == "__main__":
    main()