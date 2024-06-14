#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* node_values;

int helper(int curnode, int** graph, int n, int* charmap, int* mystack) {
    mystack[curnode] = 1;
    int maxval = 0;
    charmap[node_values[curnode-1]]++;

    if (graph[curnode][0] == 0) {
        // leaf node
        for (int i = 0; i < 26; i++) {
            maxval = (maxval > charmap[i]) ? maxval : charmap[i];
        }
        mystack[curnode] = 0;
        charmap[node_values[curnode-1]]--;
        return maxval;
    }

    for (int i = 0; i < graph[curnode][0]; i++) {
        int neigh = graph[curnode][i+1];
        if (mystack[neigh] == 1) {
            return -1;
        }
        int res = helper(neigh, graph, n, charmap, mystack);
        if (res == -1) {
            return -1;
        }
        maxval = (maxval > res) ? maxval : res;
    }

    charmap[node_values[curnode-1]]--;
    mystack[curnode] = 0;
    return maxval;
}

int maxPathValue(int n, int m, int** edges, char* values) {
    node_values = values;

    int** graph = (int**)malloc((n+1) * sizeof(int*));
    for (int i = 0; i <= n; i++) {
        graph[i] = (int*)calloc(n+1, sizeof(int));
    }

    for (int i = 0; i < m; i++) {
        int from = edges[i][0];
        int to = edges[i][1];
        graph[from][++graph[from][0]] = to;
    }

    int* charmap = (int*)calloc(26, sizeof(int));
    int maxval = 0;
    int* mystack = (int*)calloc(n+1, sizeof(int));
    for (int i = 1; i <= n; i++) {
        int res = helper(i, graph, n, charmap, mystack);
        if (res == -1) {
            return -1;
        }
        maxval = (maxval > res) ? maxval : res;
    }

    for (int i = 0; i <= n; i++) {
        free(graph[i]);
    }
    free(graph);
    free(charmap);
    free(mystack);
    return maxval;
}

int beauty_in_code(int n, int m, int* x, int* y, char* values) {
    int** edges = (int**)malloc(m * sizeof(int*));
    for (int i = 0; i < m; i++) {
        edges[i] = (int*)malloc(2 * sizeof(int));
        edges[i][0] = x[i];
        edges[i][1] = y[i];
    }
    int res = maxPathValue(n, m, edges, values);
    for (int i = 0; i < m; i++) {
        free(edges[i]);
    }
    free(edges);
    return res;
}

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    char *values = (char*)malloc(n * sizeof(char));
    int* x = (int*)malloc(m * sizeof(int));
    int* y = (int*)malloc(m * sizeof(int));
    
    for (int i = 0; i < n; i++) {
        scanf(" %c", &values[i]);
    }

    # take x input
    for (int i = 0; i < m; i++) {
        scanf("%d", &x[i]);
    }

    # take y input
    for (int i = 0; i < m; i++) {
        scanf("%d", &y[i]);
    }

}