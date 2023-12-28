import os

# folder to store figure of voxels
if not os.path.exists("NARPS_html"):
    os.mkdir("NARPS_html")

### DISPLAY results of "negative out" next to "all included data"
def generate_html_visualization_alongside_negative_out(folder_a, folder_b, output_html):
    # Get the list of files in each folder
    files_a = sorted(os.listdir(folder_a))
    files_b = sorted(os.listdir(folder_b))

    # Find common files in both folders
    common_files = set(files_a) & set(files_b)
    common_files = sorted(common_files)

    # Create HTML content
    html_content = """
    <html>
    <head>
        <title>Results Visualization alongside negative out</title>
    </head>
    <body>
        <h1>Results Visualization alongside negative out</h1>
        <h2>Right = negative out</h2>
    """

    # Add A and B visualizations with links and images for common files
    for file_name in common_files:
        file_a = os.path.join(folder_a, file_name)
        file_b = os.path.join(folder_b, file_name)

        link_a = f'<a href="../{file_a}" target="_blank">{file_name}</a>'
        img_a = f'<img src="../{file_a}" alt="{file_name}" width="100%">'

        link_b = f'<a href="../{file_b}" target="_blank">{file_name}</a>'
        img_b = f'<img src="../{file_b}" alt="{file_name}" width="100%">'

        html_content += f"""
            <div style="display: flex;">
                <div style="width: 50%;">
                    <h2>{file_name}</h2>
                    {link_a}
                    {img_a}
                </div>
                <div style="width: 50%;">
                    <h2>{file_name}</h2>
                    {link_b}
                    {img_b}
                </div>
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

for hyp in [1, 2, 5, 6, 7, 8, 9]:

    folder_path_a = 'results_in_Narps_data/hyp{}'.format(hyp)
    folder_path_b = 'results_in_Narps_data_negative_out/hyp{}'.format(hyp)
    output_html_path = 'NARPS_html/raw_brains_hyp{}.html'.format(hyp)
    generate_html_visualization_alongside_negative_out(folder_path_a, folder_path_b, output_html_path)



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


# Example usage
folder_path = '/home/jlefortb/SDMA/results_in_Narps_data/data'
output_html_path = 'NARPS_html/all_pngs_data.html'
generate_html_visualization_of_all_png(folder_path, output_html_path)

for hyp in [1, 2, 5, 6, 7, 8, 9]:
    folder_path = '/home/jlefortb/SDMA/results_in_Narps_data/hyp{}/voxel_per_team'.format(hyp)
    output_html_path = 'NARPS_html/voxel_per_team_hyp{}.html'.format(hyp)
    generate_html_visualization_of_all_png(folder_path, output_html_path, size=60)

for cluster in range(2, 10):  
    folder_path = '/home/jlefortb/SDMA/results_in_Narps_data/hyp{}/{}clusters_results'.format(1, cluster)
    output_html_path = 'NARPS_html/{}clusters_solution.html'.format(cluster)
    generate_html_visualization_of_all_png(folder_path, output_html_path, size=50)

folder_path = '/home/jlefortb/SDMA/results_in_generated_data/plot_results'
output_html_path = 'NARPS_html/simulation_results.html'
generate_html_visualization_of_all_png(folder_path, output_html_path, size=50)