from data_preprocessor import getBoundingBoxes, readDigits
import cv2
import PySimpleGUI as sg

def create_gui():
    #sg.theme_previewer()
    sg.theme('Reds')   

    layout = [
            [sg.Text("Choose a file: ", pad=(0,10)), sg.Input(), sg.FileBrowse(key="-IN-")],
            [sg.Button("Submit", pad=(0,10))], [sg.Button("Live View", pad=(0,10))]
            ]

    window = sg.Window('Reading numbers', layout, size=(800, 600), finalize=True)
    last_frame = None
    event = None

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break

        if event == "Submit":
            path = values["-IN-"]
            img = cv2.imread(path)
            coordinates_array = getBoundingBoxes(img)
            readDigits(coordinates_array, img)
            break

        if event == "Live View":
            break

    window.close()

    if event == "Live View":
        vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            _, frame = vid.read()
            last_frame = frame.copy()
            getBoundingBoxes(img = frame, visualize=True, live_view=True)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        vid.release()
        cv2.destroyAllWindows()
        coordinates_array = getBoundingBoxes(last_frame, live_view=False)
        readDigits(coordinates_array, img=last_frame)

if __name__ == '__main__':
    create_gui()