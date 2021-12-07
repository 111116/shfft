#include <iostream>
#include <cmath>
#include <ctime>
#include <vector>
using namespace std;

int main()
{
	std::vector<float> Tcoeffs;
        Tcoeffs.reserve(500000 * 9 * 9);

        int numVertex = 0;
        FILE* fp = fopen("../dragon_and_bunny_coeff.txt", "r");
        float tmpVal;
        while (fscanf(fp, "%f ", &tmpVal) != EOF)
        {
            Tcoeffs.push_back(tmpVal);
            for (int i = 0; i < 9 * 9 - 1; i++)
            {
                fscanf(fp, "%f ", &tmpVal);
                Tcoeffs.push_back(tmpVal);
            }
            numVertex++;
        }
        fclose(fp);
        printf("Numvertex = %d\n", numVertex);
        
	fp = fopen("../dragon_and_bunny_coeff14.txt", "w");
	for (uint i = 0; i < numVertex; i++)
        {
            uint idx = i * 81;
            for (uint j = 0; j < 14 * 14; j++)
	    {
		if(j < 81)
                	fprintf(fp, "%.6f ", Tcoeffs[idx + j]);
		else
			fprintf(fp, "%.6f ", Tcoeffs[idx + j % 81] * (((float)rand() / RAND_MAX) - 0.5) * 2 );
	    }
            fprintf(fp, "\n");
        }
        fclose(fp);
	return 0;
}
