from Utils import *
import matplotlib.pyplot as plt
pie_labels = ["Blues",  "Classical",    "Country",  "Disco",    "Hiphop",       "Jazz",     "Metal",        "Pop",          "Reggae",       "Rock"]
pie_colors = ['red',    'orangered',    'orange',   'gold',     'greenyellow',  'springgreen','lightskyblue', 'royalblue',    'mediumpurple', "mediumorchid"]

def main():
    load_all_model()

    while True:

        fullpath = input("Music file path: ")
        if fullpath == "exit":
            break
        if not os.path.isfile(fullpath):
            print("File does not exist.")
            continue
        print("Computing...")
        feature_res, cnn_res, combine_res = manual_test(fullpath)
        filename = os.path.basename(fullpath)

        '''
        feature_res = np.load("feature_res.npy")
        cnn_res = np.load("cnn_res.npy")
        combine_res = np.load("combine_res.npy")
        filename="FurEliseRemix"
        '''
        fig = plt.figure()
        ax = fig.add_subplot(131)
        make_pie_chart(ax, feature_res, "Dự đoán của mô hình\nANN từ 13 feature")
        ax = fig.add_subplot(132)
        make_pie_chart(ax, cnn_res, "Dự đoán của mô hình\nCNN từ mel-spectrogram")
        ax = fig.add_subplot(133)
        make_pie_chart(ax, combine_res, "Dự đoán của mô hình\nANN kết hợp")
        fig.suptitle('Kết quả dự đoán thể loại cho ' + filename, fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.8)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()


def make_autopct(pct):
    return ('%.2f' % pct) + "%" if pct > 5 else ''

def make_label(res):
    labels = []
    for i in range (10):
        if res[i] > 0.05:
            labels.append(pie_labels[i])
        else:
            labels.append("")
    return labels

def make_label_percent(res):
    labels = []
    for i in range(10):
        labels.append(pie_labels[i] + ': ' + ('%.2f' % (res[i] * 100)) + "%")
    return labels

def make_pie_chart(ax, res, title):
    labels = make_label(res)
    labels_percent = make_label_percent(res)
    patches, texts, autotexts = ax.pie(res, colors=pie_colors, labels=labels, startangle=90, autopct=make_autopct)
    ax.legend(patches, labels_percent, loc='upper center', bbox_to_anchor=(0.5, 1.22))
    ax.set_xlabel(title)
    ax.axis('equal')


if __name__ == "__main__":
    main()