import { Home as HomePage, UserInfo, MatchBreeds, BreedDetails } from '../screens'

export const routes = [ 
    {
        path: '/fur-ever-search-app/',
        exact: true,
        content: HomePage
    },
    {
        path: '/fur-ever-search-app/user-info/',
        exact: true,
        content: UserInfo
    },
    {
        path: '/fur-ever-search-app/match-breeds/',
        exact: true,
        content: MatchBreeds
    },
    {
        path: '/fur-ever-search-app/breed-details/:breedName',
        exact: true,
        content: BreedDetails
    }
]
