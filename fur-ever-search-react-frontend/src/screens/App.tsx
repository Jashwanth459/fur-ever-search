import React from 'react';
import { Route, Switch, BrowserRouter as Router } from 'react-router-dom'

import { routes } from '../routes/page_routes'

import './App.css';

function App() {
  return (
    <Router>
      <Switch>
        {routes.map((route) => (
          <Route path={route.path} component={route.content} exact={route.exact} />
        ))}
      </Switch>
    </Router>
  );
}

export default App;
