import os

# folder to store figure of voxels
if not os.path.exists("HCP_html"):
    os.mkdir("HCP_html")


### DISPLAY all png from a folder
def generate_html_visualization_of_all_png(folder_path, output_html, size=60):
    # Get the list of PNG files in the folder
    png_files = [file for file in sorted(os.listdir(folder_path)) if file.lower().endswith('.png')]

    # Create HTML content
    html_content = """
    <html>
    <head>
        <title>Results Visualization all pngs from folder</title>
    </head>
    <body>
        <h1>Results Visualization all pngs from folder</h1>
    """

    # Add visualizations with links and images for each PNG file
    for png_file in png_files:
        file_path = os.path.join(folder_path, png_file)
        title = os.path.splitext(png_file)[0]  # Get the title without the ".png" extension

        link = f'<a href="{file_path}" target="_blank">{title}</a>'
        if size == 60:
            img = f'<img src="{file_path}" alt="{title}" width="60%">'
        elif size == 70:
            img = f'<img src="{file_path}" alt="{title}" width="70%">'
        elif size == 80:
            img = f'<img src="{file_path}" alt="{title}" width="80%">'
        elif size == 90:
            img = f'<img src="{file_path}" alt="{title}" width="90%">'
        elif size == 100:
            img = f'<img src="{file_path}" alt="{title}" width="100%">'
        else:
            img = f'<img src="{file_path}" alt="{title}" width="50%">'


        html_content += f"""
            <div>
                <h2>{title}</h2>
                {link}
                {img}
            </div>
        """

    # Close HTML content
    html_content += """
    </body>
    </html>
    """

    # Write HTML content to file
    with open(output_html, 'w') as html_file:
        html_file.write(html_content)



folder_path = '/home/jlefortb/SDMA/results_in_HCP_data'
output_html_path = 'HCP_html/sdma_outputs.html'
generate_html_visualization_of_all_png(folder_path, output_html_path)

folder_path = '/home/jlefortb/SDMA/results_in_HCP_data/data'
output_html_path = 'HCP_html/raw_and_residuals_maps.html'
generate_html_visualization_of_all_png(folder_path, output_html_path)

folder_path = '/home/jlefortb/SDMA/results_in_HCP_data/voxel_per_team'
output_html_path = 'HCP_html/SDMA_in_voxel.html'
generate_html_visualization_of_all_png(folder_path, output_html_path)


folder_path = '/home/jlefortb/SDMA/results_in_HCP_data/2clusters_results'
output_html_path = 'HCP_html/SDMA_clusters.html'
generate_html_visualization_of_all_png(folder_path, output_html_path)