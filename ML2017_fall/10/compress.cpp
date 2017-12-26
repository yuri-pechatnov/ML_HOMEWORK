
#include <cstdio>
#include <cstdlib>
#include <unordered_map>

using ui32 = uint32_t;

int main() {
    char line[1000];

    FILE *input, *output;

    std::unordered_map<uint64_t, uint32_t> users;
    std::unordered_map<uint32_t, uint32_t> movies;

    std::unordered_map<uint32_t, uint32_t> movieCounts;
    std::unordered_map<uint32_t, double> movieSums;
    {
        FILE* input = fopen("data/train.csv", "rt");
        FILE* output = fopen("data/user_average.csv", "wt");

        // header
        fgets(line, sizeof(line), input);
        fputs(line, output);

        double rating;
        uint32_t mId;
        uint64_t uId;
        while (fscanf(input, "%lf,%u,%llu\n", &rating, &mId, &uId) == 3) {
            movieCounts[mId]++;
            movieSums[mId] += rating;
        }

        fclose(input);
        fclose(output);
    }


    std::unordered_map<uint32_t, uint32_t> userCounts;
    std::unordered_map<uint32_t, double> userSum;

    {
        FILE* input = fopen("data/train.csv", "rt");
        FILE* output = fopen("data/train2.csv", "wt");
        FILE* output3 = fopen("data/train3.csv", "wt");
        FILE* outputUA = fopen("data/user_average.csv", "wt");
        FILE* outputVW = fopen("data/train2vw.csv", "wt");

        // header
        fgets(line, sizeof(line), input);
        fputs(line, output);

        double rating;
        uint32_t mId;
        uint64_t uId;
        while (fscanf(input, "%lf,%u,%llu\n", &rating, &mId, &uId) == 3) {
            if (movieCounts[mId] >= 5) {
                ui32 cuId = movies.emplace(mId, movies.size()).first->second;
                ui32 cmId = users.emplace(uId, users.size()).first->second;

                userCounts[cuId]++;
                userSum[cuId] += rating;

                fprintf(output, "%0.1lf,%u,%u\n", rating, cmId, cuId);
                fprintf(output3, "%u,%u,%0.1lf\n", cuId, cmId, rating);
                fprintf(outputVW, "%0.1lf |u %u |i %u\n", rating, cuId, cmId);
            }
        }

        for (auto mp : userCounts) {
            fprintf(outputUA, "%u,%0.1lf\n", mp.first, userSum[mp.first] / mp.second);
        }

        fclose(input);
        fclose(output);
        fclose(output3);
        fclose(outputVW);
        fclose(outputUA);
    }

    {
        FILE* input = fopen("data/test.csv", "rt");
        FILE* output = fopen("data/test2.csv", "wt");

        FILE* outputWA = fopen("data/test2wa.csv", "wt");

        // header
        fgets(line, sizeof(line), input);
        fputs(line, output);

        double rating;
        uint32_t mId;
        uint64_t uId;
        while (fscanf(input, "%u,%llu\n", &mId, &uId) == 2) {
            ui32 cuId = movies.emplace(mId, movies.size()).first->second;
            ui32 cmId = users.emplace(uId, users.size()).first->second;
            fprintf(output, "%u,%u\n", cmId, cuId);
            fprintf(outputWA, "%u,%u,%0.4lf\n", cmId, cuId, movieSums[mId] / movieCounts[mId]);
        }

        fclose(input);
        fclose(output);

        fclose(outputWA);
    }


}

