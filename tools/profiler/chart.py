import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def timeline_chart(result):
    max_iter_time = 0.0
    for host in result:
        max_iter_time = max(max_iter_time, host["avg_iter_time_ms"])

    fig, ax = plt.subplots()
    color_options = ['steelblue', 'mediumorchid', 'yellowgreen', 'lightcoral']

    y_label_font_size = 8
    figure_width_inches = 11

    stream_num = 0
    h_between_stream = 7
    addtion_base_h = 1
    stream_base_h = 10
    h_val_inches_ratio = 0.05

    current_base_h_pos = h_between_stream
    y_labels = []
    y_ticks = []

    for host in result:
        for device_id, streams in host["timeline"].items():
            stream_num += len(streams)
            for stream_name, timeline in streams.items():
                max_height_stream_bar = 0
                color_idx = 0
                overlapped_check = []
                for event in timeline:
                    occupied_h = []
                    for idx, (end, h) in enumerate(overlapped_check):
                        if end > event["avg_iter_start_to_event_start_time_ms"]:
                            # overlaped
                            occupied_h.append(h)
                    h = 0
                    while h in occupied_h:
                        h += 1
                    height = stream_base_h + addtion_base_h * h
                    max_height_stream_bar = max(height, max_height_stream_bar)
                    overlapped_check.append(
                        (event["avg_iter_start_to_event_start_time_ms"] + event["avg_measured_time_ms"], h)
                    )

                    label = "{}\navg_measured_time_ms: {}\navg_iter_start_to_event_start_time_ms: {}".format(
                        event["label"], event['avg_measured_time_ms'], event["avg_iter_start_to_event_start_time_ms"]
                    )

                    ax.add_patch(
                        patches.Rectangle(
                            xy=(event["avg_iter_start_to_event_start_time_ms"], current_base_h_pos),  # point of origin.
                            width=event["avg_measured_time_ms"],
                            height=height,
                            color=color_options[color_idx % len(color_options)],
                            fill=True,
                            alpha=0.6,
                            label=label
                        )
                    )
                    color_idx += 1

                y_ticks.append(current_base_h_pos + float(max_height_stream_bar) / 2)
                y_labels.append(
                    host["host_name"] + '_'
                    + 'd' + device_id.split('_')[-1] + '_'
                    + 's' + stream_name.split('_')[-1])
                current_base_h_pos += max_height_stream_bar + h_between_stream

    ax.add_patch(
        patches.Rectangle(
            xy=(0, current_base_h_pos),  # point of origin.
            width=max_iter_time,
            height=height,
            color='steelblue',
            fill=True,
            alpha=0.9,
            label="Whole Iter: {}".format(max_iter_time)
        )
    )

    annot = ax.annotate("", xy=(0,0), xytext=(0, 35), textcoords="offset points",
                        arrowprops=dict(arrowstyle="->"), fontsize=8, ha='center', va='center')
    annot.set_visible(False)

    y_ticks.append(current_base_h_pos + float(max_height_stream_bar) / 2)
    y_labels.append("Whole Iteration")
    current_base_h_pos += max_height_stream_bar + h_between_stream

    def click(event):
        min_distance = 9999999.0
        min_idx = -1
        for idx, p in enumerate(ax.patches):
            p.set_hatch(None)
            if p.contains(event)[0]:
                if event.xdata - p.get_x() < min_distance:
                    min_idx = idx
                    min_distance = event.xdata - p.get_x()

        if min_idx >= 0:
            x, y = ax.patches[min_idx].get_xy()
            annot.xy = (x + ax.patches[min_idx].get_width() / 2.0, y + ax.patches[min_idx].get_height() / 2.0)
            annot.set_text(ax.patches[min_idx].get_label())
            ax.patches[min_idx].set_hatch('/')
            annot.set_visible(True)
        else:
            annot.set_visible(False)
        fig.canvas.draw_idle()

    # add callback for mouse click
    fig.canvas.mpl_connect('button_press_event', click)
    plt.xlabel("In Milliseconds")
    fig.set_size_inches(figure_width_inches, current_base_h_pos * h_val_inches_ratio)
    ax.set_xlim(0, max_iter_time)
    ax.set_ylim(0, current_base_h_pos)
    ax.grid(True)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontdict={ 'fontsize' : y_label_font_size })

    plt.show()


def scaling_chart(results, names):
    bar_width = 0.2
    space_between_result = 0.1
    x_label_font_size = 8
    names_font_size = 9
    color_options = ['steelblue', 'mediumorchid', 'yellowgreen', 'lightcoral']
    fig, ax = plt.subplots()

    x_ticks = []
    x_labels = []
    current_start_x = space_between_result
    max_h = 0.0
    for i, result in enumerate(results):
        iter_times = [host["avg_iter_time_ms"] for host in result]
        iter_times = max(iter_times)
        max_h = max(iter_times, max_h)
        labels = {}
        for host in result:
            for device in host["timeline"].keys():
                for stream in host["timeline"][device].keys():
                    for event in host["timeline"][device][stream]:
                        if event['label'] not in labels.keys():
                            labels[event['label']] = []
                        labels[event['label']].append(event['avg_measured_time_ms'])

        ax.add_patch(
            patches.Rectangle(
                xy=(current_start_x, 0),  # point of origin.
                width=bar_width,
                height=iter_times,
                color='steelblue',
                fill=True,
                alpha=0.9,
                label="Whole Iter: {}".format(iter_times)
            )
        )
        color_idx = 0
        current_height_avg = 0.0
        current_height_max = 0.0
        for label, times in labels.items():
            avg_time = sum(times) / len(times)
            max_time = max(times)
            ax.add_patch(
                patches.Rectangle(
                    xy=(current_start_x + bar_width, current_height_avg),  # point of origin.
                    width=bar_width,
                    height=avg_time,
                    color=color_options[color_idx % len(color_options)],
                    fill=True,
                    alpha=0.6,
                    label="{}\n{} ms".format(label, avg_time)
                )
            )
            ax.add_patch(
                patches.Rectangle(
                    xy=(current_start_x + 2 * bar_width, current_height_max),  # point of origin.
                    width=bar_width,
                    height=max_time,
                    color=color_options[color_idx % len(color_options)],
                    fill=True,
                    alpha=0.6,
                    label="{}\n{} ms".format(label, max_time)
                )
            )

            max_h = max([current_height_avg, current_height_max, max_h])
            current_height_avg += avg_time
            current_height_max += max_time
            color_idx += 1

        ax.annotate(names[i], xy=(current_start_x + 3 * bar_width / 2, 0), xytext=(0, -10),
                    textcoords="offset points", fontsize=names_font_size, ha='center', va='center')
        current_start_x += 3 * bar_width + space_between_result

    ax.annotate("Left bar: Whole iter time\nMid bar: Avg time\nRight bar: Max time",
                 xy=(current_start_x / 2, 0), xytext=(0, -40),
            textcoords="offset points", fontsize=names_font_size, ha='center', va='center')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontdict={ 'fontsize' : x_label_font_size })
    plt.ylabel("In Milliseconds")

    click_annot = ax.annotate("", xy=(0,0), xytext=(15, 35), textcoords="offset points",
                        arrowprops=dict(arrowstyle="->"), fontsize=8, ha='center', va='center')
    click_annot.set_visible(False)

    def click(event):
        find = None
        for bar in ax.patches:
            bar.set_hatch(None)
            if bar.contains(event)[0]:
                find = bar
        if find:
            x, y = find.get_xy()
            click_annot.xy = (x + find.get_width() / 2.0, y + find.get_height() / 2.0)
            click_annot.set_text(find.get_label())
            find.set_hatch('/')
            click_annot.set_visible(True)
        else:
            click_annot.set_visible(False)
        fig.canvas.draw_idle()

    # add callback for mouse click
    fig.canvas.mpl_connect('button_press_event', click)

    ax.set_xlim(0, current_start_x)
    ax.set_ylim(0, max_h)
    ax.grid(True)
    fig.set_size_inches(10, 10)
    plt.show()