
# Custom query description class (unprocessed query)
class Query_desc:

    def __init__(self, edges, target):
        self.edges = edges
        self.target = target

# Custom query class
class Query:

    def __init__(self, edge_index, edge_type, target):
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.target = target
        self.embedding = None



# Creating 3 custom queries
# index 16 represents variables and index 17 represents target!


### Query 0 ### 
### 3 nodes, 2 edges ###
### node 0 being the root, node 1 being variable, node 2 = target ###

query0_description = Query_desc(
                        edges =
                        [[0 , 1, 16],
                        [16, 3, 17]],
                        target = 2)


### Query 1 ### 
### 3 nodes, 2 edges ###
### node 5 being the root, node 4 being variable, node 6 = target ###

query1_description =  Query_desc(
                        edges =
                        [[5 , 5, 16],
                        [16, 4, 17]],
                        target = 6)
    

### Query 2 ### 
### 4 nodes, 3 edges ###
### node 12 and 11 being the root, node 9 being variable, node 8 = target ###

query2_description =  Query_desc(
                        edges =
                        [[12, 1, 9],
                        [11, 1, 9],
                        [9,  1, 8]],
                        target = 8)

query_descriptions = (query0_description,query1_description,query2_description)