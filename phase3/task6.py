from appJar import gui
from task3 import get_t_closest_gestures
from task4 import get_updated_gestures
from task5 import get_updated_gestures_task5, initial_result_task5
import argparse

app = None
t = None
feedback_type = None
query_gesture = None
gestures_map= {}
prob_vectors_path = None
ppr_vectors_path = None

def submit_feedback():
    global app
    global t
    global query_gesture
    global prob_vectors_path
    global ppr_vectors_path

    relevant_gestures = []
    irrelevant_gestures = []
    all_results = []

    for key, value in app.getAllCheckBoxes().items():
        if value:
            if "irrelevant" in key:
                irrelevant_gestures.append(gestures_map[key])
            else:
                relevant_gestures.append(gestures_map[key])
        all_results.append(gestures_map[key])

    print(relevant_gestures)
    print(irrelevant_gestures)
    updated_results = []
    if feedback_type == "Probabilistic Feedback":
        print("Calling Probabilistic Feedback")
        updated_results = get_updated_gestures(relevant_gestures, irrelevant_gestures, int(t), all_results, prob_vectors_path)
    else:
        print("Calling PPR Feedback")
        updated_results = get_updated_gestures_task5(relevant_gestures, irrelevant_gestures, int(t), query_gesture)

    app.stop()
    app = gui("Query interface",handleArgs=False)
    app.addLabel("l1", "Updated Query results")
    app.getLabelWidget("l1").config(font=("Comic Sans", "30", "normal"))
    app.setSize(500, 500)
    app.setFont(20)
    app.startScrollPane("Scroll Pane")
    app.startFrame("Relevant", row=0, column=0)
    app.addLabel("Select Relevant")
    for i in range(len(updated_results)):
        app.addNamedCheckBox(updated_results[i], "relevant" + str(i))
        gestures_map["relevant"+str(i)] = updated_results[i]
    app.stopFrame()
    app.startFrame("Irrelevant", row=0, column=1)
    app.addLabel("Select Irrelevant")
    for i in range(len(updated_results)):
        app.addNamedCheckBox(updated_results[i], "irrelevant" + str(i))
        gestures_map["irrelevant"+str(i)] = updated_results[i]
    app.stopFrame()
    app.stopScrollPane()
    app.addButton("SUBMIT FEEDBACK", submit_feedback)
    app.go()


def search():
    global app
    global t
    global feedback_type
    global query_gesture
    global prob_vectors_path
    global ppr_vectors_path

    query_gesture = app.getEntry("Enter the query gesture")
    query_gesture = query_gesture + "_words.csv"

    t = app.getEntry("Enter the number of results to be returned")
    feedback_type = app.getRadioButton("relevance_feedback_type")

    initial_search_results = []
    if feedback_type == "Probabilistic Feedback":
        print("Calling Probabilistic Feedback")
        initial_search_results = get_t_closest_gestures(6, 3, prob_vectors_path, int(t), query_gesture)
    else:
        print("Calling PPR Feedback")
        initial_search_results = initial_result_task5(ppr_vectors_path, int(t), query_gesture)
 
    app.stop()
    app = gui("Query interface",handleArgs=False)
    app.addLabel("l1", "Query results")
    app.getLabelWidget("l1").config(font=("Comic Sans", "30", "normal"))
    app.setSize(500, 500)
    app.setFont(20)

    app.startScrollPane("Scroll Pane")
    app.startFrame("Relevant", row=0, column=0)
    app.addLabel("Select Relevant")
    for i in range(len(initial_search_results)):
        app.addNamedCheckBox(initial_search_results[i], "relevant" + str(i))
        gestures_map["relevant"+str(i)] = initial_search_results[i]
    app.stopFrame()
    app.startFrame("Irrelevant", row=0, column=1)
    app.addLabel("Select Irrelevant")
    for i in range(len(initial_search_results)):
        app.addNamedCheckBox(initial_search_results[i], "irrelevant" + str(i))
        gestures_map["irrelevant"+str(i)] = initial_search_results[i]
    app.stopFrame()
    app.stopScrollPane()
    app.addButton("SUBMIT FEEDBACK", submit_feedback)
    app.go()


parser = argparse.ArgumentParser(description='Task6')
parser.add_argument('--prob_vectors', help='input vectors path', required=True)
parser.add_argument('--ppr_vectors', help='input vectors path', required=True)
args = parser.parse_args()
prob_vectors_path = args.prob_vectors
ppr_vectors_path = args.ppr_vectors

# create a GUI variable called app
app = gui("Query Interface", handleArgs=False)

app.addLabelEntry("Enter the query gesture")
app.addLabelEntry("Enter the number of results to be returned")
app.setFont(18)
app.addRadioButton("relevance_feedback_type", "Probabilistic Feedback")
app.addRadioButton("relevance_feedback_type", "PPR Feedback")
app.addButton("search", search)
app.go()




