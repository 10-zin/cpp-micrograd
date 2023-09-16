#include <stdio.h>
#include <stdlib.h>

#define MAX_ENTRIES 30

typedef struct {
    float number;
    float label;  // 0 for even, 1 for odd
} Entry;

Entry* load_data() {
    FILE *fp;
    Entry *entries = malloc(MAX_ENTRIES * sizeof(Entry));
    if (entries == NULL) {
        perror("Failed to allocate memory");
        exit(1);
    }
    int count = 0;

    fp = fopen("data.txt", "r");
    if (fp == NULL) {
        perror("Failed to open file");
        free(entries);
        exit(1);
    }

    while (fscanf(fp, "%f %f", &entries[count].number, &entries[count].label) != EOF && count < MAX_ENTRIES) {
        count++;
    }
    fclose(fp);

    return entries;
}

// int main() {
//     Entry* entries = load_data();
    
//     for (int i = 0; i < MAX_ENTRIES; i++) {
//         printf("Number: %d, Label: %d\n", entries[i].number, entries[i].label);
//     }

//     free(entries);  // Remember to free the memory when done
//     return 0;
// }
