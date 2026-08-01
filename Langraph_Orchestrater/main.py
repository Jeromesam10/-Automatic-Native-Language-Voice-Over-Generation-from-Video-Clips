from graph import graph


initial_state = {
    "input": "Describe a traffic video"
}

result = graph.invoke(initial_state)

print("\n==============================")
print("FINAL STATE")
print("==============================")

print(result)