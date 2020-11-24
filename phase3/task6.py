# import the library
from appJar import gui
from task3 import get_t_closest_gestures
from task4 import get_updated_gestures
from task5 import get_updated_gestures_task5, initial_result_task5

app = None
t = None
feedback_type = None
query_gesture = None
def submit_feedback():
    global app
    global t
    global query_gesture
    relevant_gestures = []
    irrelevant_gestures = []
    all_results = []
    for key, value in app.getAllCheckBoxes().items():
        if value:
            relevant_gestures.append(key)
        else:
            irrelevant_gestures.append(key)
        all_results.append(key)
    updated_results = []
    if feedback_type == "Probabilistic Feedback":
        print("Calling Probabilistic Feedback")
        updated_results = get_updated_gestures(relevant_gestures, int(t), all_results)
    else:
        print("Calling PPR Feedback")
        updated_results = get_updated_gestures_task5(relevant_gestures, irrelevant_gestures, int(t), query_gesture)
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
    global feedback_type
    global query_gesture
    query_gesture = app.getEntry("Enter the gesture")
    query_gesture = query_gesture + "_words.csv"

    t = app.getEntry("Enter the number of results to be returned")


    initial_search_results = initial_result_task5("../outputs/tf_idf_pca_vectors.csv", int(t), query_gesture)
    feedback_type = app.getRadioButton("relevance_feedback_type")
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
app.addRadioButton("relevance_feedback_type", "Probabilistic Feedback")
app.addRadioButton("relevance_feedback_type", "PPR Feedback")
app.addButton("search", search)
app.go()




