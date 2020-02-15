#include <iostream>
#include <cstring>

using namespace std;

struct Samurai{
	char name[40];
	int numberOfCompletedTasks;
	bool experienced;
	char sex;
	bool ninja;
};

void checkExperience(Samurai *samurai){
	((*samurai).numberOfCompletedTasks > 15) ? (*samurai).experienced = true : (*samurai).experienced = false;
}

void showSuitabilityForTask(char s, bool ninja){
	if(s == 'm'){

		if(ninja) cout << "Suitable for sabotage" << endl;
		else cout << "Suitable for field battle" << endl;

	}else if(s == 'f'){

		if(ninja) cout << "Suitable for infiltration" << endl;
		else cout << "Suitable for diplomacy" << endl;

	}else cout << "Char not accepted. Accepted are only the values 'm' for male and 'f' for female." << endl;
}

bool showExperiencedSamurai(Samurai samurai){
	(samurai.numberOfCompletedTasks > 15) ? cout << "Experienced" : cout << "Inexperienced" << endl;

	return samurai.experienced;
}


int main(){
	Samurai samurai;

	strcpy(samurai.name, "Kusunoki Masashige");
	samurai.numberOfCompletedTasks = 34;
	samurai.ninja = 0;
	samurai.sex = 'm';

	checkExperience(&samurai);
	bool e = showExperiencedSamurai(samurai);
	cout << endl;

	showSuitabilityForTask(samurai.sex, samurai.ninja);
	cout << endl;

	if(samurai.experienced == e) cout << "Correct\n";
	else cout << "Incorrect\n";

	return 0;
}
