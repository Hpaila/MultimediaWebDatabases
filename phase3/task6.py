# import the library
from appJar import gui
from task3 import get_t_closest_gestures
from task4 import get_updated_gestures

app = None
t = None
def submit_feedback():
    global app
    global t
    relevant_gestures = []
    all_results = []
    for key, value in app.getAllCheckBoxes().items():
        if value:
            relevant_gestures.append(key)
        all_results.append(key)
    updated_results = get_updated_gestures(relevant_gestures, int(t), all_results)
    app.stop()
    app = gui("Query interface")
    app.addLabel("l1", "Updated Query results")
    app.getLabelWidget("l1").config(font=("Comic Sans", "30", "normal"))
    app.setSize(500, 500)
    app.startScrollPane("Scroll Pane")
    for res in updated_results:
        app.addCheckBox(res)
    
    app.setFont(20)
    app.stopScrollPane()
    app.addButton("SUBMIT FEEDBACK", submit_feedback)
    app.go()


def search():
    global app
    global t
    query_gesture = app.getEntry("Enter the gesture")
    query_gesture = query_gesture + "_words.csv"

    t = app.getEntry("Enter the number of results to be returned")

    initial_search_results = get_t_closest_gestures(6, 3, "outputs/vectors/tf_idf_vectors.csv", int(t), query_gesture)

    app.stop()
    app = gui("Query interface")
    app.addLabel("l1", "Query results")
    app.getLabelWidget("l1").config(font=("Comic Sans", "30", "normal"))
    app.startScrollPane("Scroll Pane")
    app.setSize(500, 500)
    for res in initial_search_results:
        app.addCheckBox(res)
    app.setFont(20)
    app.stopScrollPane()
    app.addButton("SUBMIT FEEDBACK", submit_feedback)
    app.go()

# create a GUI variable called app
app = gui("Query interface")

app.addLabelEntry("Enter the gesture")
app.addLabelEntry("Enter the number of results to be returned")
app.setFont(18)

app.addButton("search", search)
app.go()




