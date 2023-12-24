import { Home as HomePage, UserInfo, MatchBreeds, BreedDetails } from '../screens'

export const routes = [ 
    {
        path: '/fur-ever-search/',
        exact: true,
        content: HomePage
    },
    {
        path: '/user-info/',
        exact: true,
        content: UserInfo
    },
    {
        path: '/match-breeds/',
        exact: true,
        content: MatchBreeds
    },
    {
        path: '/breed-details/:breedName',
        exact: true,
        content: BreedDetails
    }
]
