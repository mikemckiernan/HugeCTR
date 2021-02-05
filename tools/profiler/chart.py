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
                if event.ydata - p.get_x() < min_distance:
                    min_idx = idx
                    min_distance = event.ydata - p.get_x()

        if min_idx > 0:
            x, y = ax.patches[min_idx].get_xy()
            annot.xy = (x + ax.patches[min_idx].get_width() / 2.0, y + ax.patches[min_idx].get_height() / 2.0)
            annot.set_text(ax.patches[min_idx].get_label())
            ax.patches[min_idx].set_hatch('/')
            annot.set_visible(True)
        else:
            annot.set_visible(False)
        fig.canvas.draw_idle()

    # add callback for mouse moves
    fig.canvas.mpl_connect('button_press_event', click)

    fig.set_size_inches(figure_width_inches, current_base_h_pos * h_val_inches_ratio)
    ax.set_xlim(0, max_iter_time)
    ax.set_ylim(0, current_base_h_pos)
    ax.grid(True)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontdict={ 'fontsize' : y_label_font_size })

    plt.show()


def scaling_chart(results, names):
    bar_width = 0.2
    x_label_font_size = 7
    names_font_size = 9
    fig, ax = plt.subplots()

    iter_time_data = []
    label_time_avg_data = []
    label_time_max_data = []

    x_axis = np.array([i + 1 for i in range(len(results))])
    for result in results:
        iter_times = [host["avg_iter_time_ms"] for host in result]
        avg_iter_times = sum(iter_times) / len(iter_times)
        iter_time_data.append(avg_iter_times)

        labels = {}
        for host in result:
            for device in host["timeline"].keys():
                for stream in host["timeline"][device].keys():
                    for event in host["timeline"][device][stream]:
                        if event['label'] not in labels.keys():
                            labels[event['label']] = []
                        labels[event['label']].append(event['avg_measured_time_ms'])

        label_time_avg_data_this_result = []
        label_time_max_data_this_result = []
        for _, times in labels.items():
            label_time_avg_data_this_result.append(sum(times) / len(times))
            label_time_max_data_this_result.append(max(times))
        label_time_avg_data.append(label_time_avg_data_this_result)
        label_time_max_data.append(label_time_max_data_this_result)

    iter_time_data = np.array(iter_time_data)
    label_time_avg_data = np.array(label_time_avg_data)
    label_time_max_data = np.array(label_time_max_data)

    plt.bar(x_axis, iter_time_data, bar_width, align='edge')
    accumulate_label_time_avg_data = np.zeros_like(label_time_avg_data[:, 0])
    accumulate_label_time_max_data = np.zeros_like(label_time_max_data[:, 0])

    for i in range(label_time_avg_data.shape[1]):
        plt.bar(x_axis + bar_width, label_time_avg_data[:, i], bar_width,
                bottom=accumulate_label_time_avg_data, align='edge')
        plt.bar(x_axis + 2 * bar_width, label_time_max_data[:, i], bar_width,
                bottom=accumulate_label_time_max_data, align='edge')
        accumulate_label_time_avg_data += label_time_avg_data[:, i]
        accumulate_label_time_max_data += label_time_max_data[:, i]

    x_ticks = np.concatenate((x_axis + bar_width / 2, x_axis + bar_width + bar_width / 2,
                             x_axis + 2 * bar_width + bar_width / 2))
    x_labels = ['Whole iteration', 'Avg_mesured_ms', 'Max_measured_ms'] * len(results)
    print(x_ticks)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontdict={ 'fontsize' : x_label_font_size })

    for i in range(len(results)):
        annot = ax.annotate(names[i], xy=(i + 1 + 3 * bar_width / 2, 0), xytext=(0, -30), 
                            textcoords="offset points", fontsize=names_font_size, ha='center', va='center')

    ax.grid(True)
    fig.set_size_inches(8, 10)
    plt.show()