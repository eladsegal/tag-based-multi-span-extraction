import React from 'react';
import {
    Collapse,
    Navbar,
    NavbarToggler,
    NavbarBrand,
  } from 'reactstrap';
import { shouldUpdate } from '../utils';
import ExplorerSettings from './explorer-settings/ExplorerSettings';
import ExplorerTable from './ExplorerTable';

const updateSignals = ['dataset', 'expandAll', 'filteredAnswerTypes', 'navbarSticky', 'isOpen']
class DatasetExplorer extends React.Component {
    constructor(props) {
        super(props);

        this.settingsChange = this.settingsChange.bind(this);
        this.getClearSelectedAnswersFunc = this.getClearSelectedAnswersFunc.bind(this);
        this.toggle = this.toggle.bind(this);
        this.state = {
            useLocalDataset: false,
            expandAll: false,
            filteredAnswerTypes: ['multi_span', 'single_span', 'number', 'date'],
            // ^ settings available for configuration

            dataset: undefined,
            clearSelectedAnswersFunc: undefined,

            navbarSticky: 'top',
            isOpen: false
        };
    }

    shouldComponentUpdate(nextProps, nextState) {
        return shouldUpdate(updateSignals, this.props, this.state, nextProps, nextState)
    }

    settingsChange(settings) {
        this.setState({
            useLocalDataset: settings.useLocalDataset,
            expandAll: settings.expandAll,
            filteredAnswerTypes: settings.filteredAnswerTypes,
            dataset: settings.dataset
        });
    }
    
    getClearSelectedAnswersFunc(func) {
        this.setState({ clearSelectedAnswersFunc: func })
    }

    toggle() {
        this.setState({
          isOpen: !this.state.isOpen
        });
    }

    render() {
        return <div>
                    <Navbar color="light" light expand="sm" sticky={this.state.navbarSticky}>
                        <NavbarBrand onClick={() => {this.setState({ navbarSticky: this.state.navbarSticky === 'top' ? undefined : 'top' })}}>DROP Explorer</NavbarBrand>
                        <NavbarToggler onClick={this.toggle} />
                        <Collapse isOpen={this.state.isOpen} navbar>
                            <ExplorerSettings onChange={this.settingsChange}
                                useLocalDataset={this.state.useLocalDataset}
                                expandAll={this.state.expandAll}
                                filteredAnswerTypes={this.state.filteredAnswerTypes}
                                clearSelectedAnswersFunc={this.state.clearSelectedAnswersFunc} />
                        </Collapse>
                    </Navbar>
                    <ExplorerTable dataset={this.state.dataset} 
                        expandAll={this.state.expandAll} 
                        filteredAnswerTypes={this.state.filteredAnswerTypes} 
                        sendClearSelectedAnswersFunc={this.getClearSelectedAnswersFunc}/>
                </div>
    }
}

export default DatasetExplorer;