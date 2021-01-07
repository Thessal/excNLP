# Generate report
from reporter import topic_visualizer
import os


def report_to_file(document, output_path="reports"):
    reports = {name: {"markdown":""} for name in document.names.values()}

    ## Generate plots
    for file in document.iter_file(unit="paragraph"):
        topic_kde, topic_segment = topic_visualizer.visualize(file)

        name = file["name"]
        report =  reports[name]
        report["plots"] = {
            "topic_kde": topic_kde,
            "topic_segment": topic_segment,
        }
        for plot_name in report["plots"]:
            report["markdown"] += f"#### {plot_name}\n"
            report["markdown"] += f"![{plot_name}]({name}_{plot_name}.png)\n\n"


    ## Write files
    for name in reports.keys():
        report = reports[name]
        plots = report["plots"]
        for plot_name in plots:
            with open(os.path.join(output_path, f"{name}_{plot_name}.png"), 'wb') as out:
                out.write(plots[plot_name].getbuffer())
        with open(os.path.join(output_path, f"{name}.md"), 'w') as out:
            out.write(report["markdown"])
