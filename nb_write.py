import nbformat
from datetime import datetime
import base64
import io

from nbformat.v4 import new_notebook
from nbformat.v4 import new_markdown_cell
from nbformat.v4 import new_code_cell
from nbformat.v4 import new_output

class JupNotebookWriter:
    def __init__(self, notebook_name):
        now = datetime.now()
        self.filename = notebook_name + '_' + now.strftime("%Y-%m-%d-%H%M%S") + '.ipynb'
        self.curr_time = now
        self.notebook = new_notebook()
        self.notebook['cells'] = []
        self.obj_counter = 0
        
        heading = "%s (Execution Time: %s)" % (notebook_name, now.strftime("%b %d %Y, %I:%M:%S %p %Z"))
        self.write_heading_cell(heading)
    # End fn __init__
    
    def write_heading_cell(self, heading, level=1):
        prefixes = ["", "#", "##", "###", "####", "#####", "######"]
        self.notebook['cells'].append(new_markdown_cell(source='%s %s' % (prefixes[level], heading)))
    # End fn write_heading_cell
    
    def write_text_cell(self, text):
        self.notebook['cells'].append(new_markdown_cell(source=text))
    # End fn write_text_cell
    
    def write_image_cell(self, image_data, caption):
        self.obj_counter = self.obj_counter + 1
        img_name = 'img' + str(self.obj_counter)
        
        b64_enc_image_data = base64.b64encode(image_data.getvalue()).decode()
        md_source = '%s\n![%s](attachment:%s)' % (caption, img_name, img_name)
        md_attachments = {img_name: {'image/png': b64_enc_image_data}}
        
        self.notebook['cells'].append(new_markdown_cell(source=md_source, attachments=md_attachments))
    # End fn write_image_cell
    
    def write_plot_cell(self, plot, caption):
        image_data = io.BytesIO()
        plot.savefig(image_data, format='png')
        plot.clf() # Clear figure
        self.write_image_cell(image_data, caption)
    # End fn write_plot_cell
    
    def finalize(self):
        nbformat.write(self.notebook, './notebooks/' + self.filename)
# End class JupNotebookWriter

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    
    jnw = JupNotebookWriter('test_notebook')
    jnw.write_heading_cell('Test Section 1', level=2)
    jnw.write_text_cell('Some text with *markdown* formatting.')

    # Plot and write a sine wave
    x = np.arange(0, 10, 0.1)
    plt.plot(x, np.sin(x))
    plt.title('Sine wave')
    plt.xlabel('X')
    plt.ylabel('Y = sin(X)')
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    jnw.write_plot_cell(plt, 'Test plot image 1 (sine wave)')
    
    jnw.finalize()
# End if
